"""Reusable Gym lifecycle for CarMaker environments."""
from abc import ABC, abstractmethod
from enum import Enum
import logging

import gym
from parametrization_testrun import (
    cmapi, make_variations, DEFAULT_TERMINATION_TIME, termination_time_seconds,
)
import asyncio
import nest_asyncio  # Import nest_asyncio

logger = logging.getLogger(__name__)


class SimulatorState(Enum):
    CLOSED = "closed"
    STARTING = "starting"
    PAUSED = "paused"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class CarMakerEnv(gym.Env, ABC):
    def __init__(self, termination_time=DEFAULT_TERMINATION_TIME):
        super().__init__()

        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()

        # Use the same duration for the RL episode and TestRun variation.
        self.dt = 0.0
        self.time_step = 0.02
        self.maxTime = termination_time_seconds(termination_time)
        self.sync_timeout = 30.0
        self.simcontrol = None
        self.state = SimulatorState.CLOSED
        self._simulator_active = False
        self._simulation_started = False
        self._episode_end = None
        self._simulation_error = None
        self._last_observation = None
        self.loop = asyncio.get_event_loop()  # Create an event loop
        print("Init success")

    async def _bounded(self, operation):
        return await asyncio.wait_for(operation, timeout=self.sync_timeout)

    async def _recover_failure(self):
        """Discard failed sessions without replacing the triggering exception."""
        try:
            await self.stop_simulation()
        except BaseException:
            logger.warning("Simulator cleanup failed during recovery", exc_info=True)
        finally:
            self.simcontrol = None
            self._simulator_active = False
            self._simulation_started = False
            self.state = SimulatorState.FAILED

    async def start_simulation(self):
        # A new session always owns a fresh controller.
        await self.stop_simulation()
        self.state = SimulatorState.STARTING
        try:
            variations = await self._bounded(make_variations(termination_time=self.maxTime))
            self.variation = variations[0]
            self.simcontrol = cmapi.SimControlInteractive()
            # Own cleanup before any controller configuration can fail.
            self._simulator_active = True
            self.simcontrol.set_variation(self.variation)
            master = cmapi.CarMaker()
            await self._bounded(self.simcontrol.set_master(master))
            await self._bounded(self.simcontrol.start_and_connect())
            self._simulation_started = True
            await self._bounded(self.simcontrol.start_sim())
            await self.pause_simulation()
        except BaseException:
            await self._recover_failure()
            raise

    async def stop_simulation(self):
        """Bound each shutdown stage and discard the controller exactly once."""
        simcontrol = self.simcontrol
        if simcontrol is None or not self._simulator_active:
            self.simcontrol = None
            return
        self.state = SimulatorState.STOPPING
        self.simcontrol = None
        self._simulator_active = False
        simulation_started = self._simulation_started
        self._simulation_started = False
        first_error = None
        try:
            if simulation_started:
                try:
                    await self._bounded(simcontrol.stop_sim())
                    await self._bounded(simcontrol.create_simstate_condition(
                        cmapi.ConditionSimState.finished
                    ).wait())
                except BaseException as exc:
                    first_error = exc
            try:
                await self._bounded(simcontrol.stop_and_disconnect())
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    logger.warning("Simulator shutdown also failed", exc_info=True)
                # stop_and_disconnect may fail before its disconnect stage.
                try:
                    await self._bounded(simcontrol.disconnect())
                except BaseException:
                    logger.warning("Fallback simulator disconnect failed", exc_info=True)
        finally:
            self.state = SimulatorState.FAILED if first_error else SimulatorState.CLOSED
        if first_error is not None:
            raise first_error

    async def pause_simulation(self):
        """Pause and confirm acknowledgement before reading or applying actions."""
        self.simcontrol.pause_sim()

        async def wait_for_pause():
            while True:
                if await self._check_simulation_end():
                    return
                acceleration, = await self.simcontrol.simio.dva_read_async("SC.TAccel")
                if acceleration == 0.0:
                    return
                await asyncio.sleep(0.001)

        await self._bounded(wait_for_pause())
        self.state = SimulatorState.PAUSED

    async def _check_simulation_end(self):
        """Inspect reported errors before classifying a completed TestRun."""
        try:
            await self.simcontrol.raise_sim_errors()
        except cmapi.SimulationError as exc:
            self._episode_end = "simulation_failure"
            self._simulation_error = str(exc)
        simstate = self.simcontrol.get_simstate().get_condition_simstate()
        if simstate in (cmapi.ConditionSimState.finished, cmapi.ConditionSimState.idle):
            self._simulation_started = False
            if self._episode_end is None:
                self._episode_end = "simulation_finished"
        return self._episode_end is not None

    async def _wait_for_progress_or_end(self, condition):
        async def monitor():
            while not await self._check_simulation_end():
                await asyncio.sleep(0.001)

        tasks = [asyncio.ensure_future(condition.wait()),
                 asyncio.ensure_future(monitor())]
        try:
            completed, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                try:
                    task.result()
                except cmapi.SimulationError as exc:
                    self._episode_end = "simulation_failure"
                    self._simulation_error = str(exc)
            await self._check_simulation_end()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def read_simulation_time(self):
        self.dt, = await self._bounded(self.simcontrol.simio.dva_read_async("Time"))

    async def advance_simulation(self):
        """Advance from the acknowledged simulator time, then pause again."""
        if await self._bounded(self._check_simulation_end()):
            return
        await self.read_simulation_time()
        target_time = min(self.dt + self.time_step, self.maxTime)
        condition = self.simcontrol.create_quantity_condition(
            lambda simulation_time: simulation_time >= target_time, "Time"
        )
        try:
            self.simcontrol.resume_sim()
            self.state = SimulatorState.RUNNING
            await self._bounded(self._wait_for_progress_or_end(condition))
        except BaseException:
            # Pausing is best effort; its failure must not hide the wait error.
            try:
                await self.pause_simulation()
            except BaseException:
                logger.warning("Simulator pause failed during recovery", exc_info=True)
            raise
        if not self._episode_end:
            await self.pause_simulation()
        try:
            await self.read_simulation_time()
        except Exception:
            if not self._episode_end:
                raise

    @abstractmethod
    async def set_action(self, action):
        """Apply an action while paused; the base class advances simulation."""
        raise NotImplementedError

    @abstractmethod
    async def get_observation(self):
        """Return an observation matching the subclass observation space."""
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self, observation):
        """Return the reward for the current observation."""
        raise NotImplementedError

    def is_done(self, observation):
        """End at the time limit; subclasses may add scenario conditions."""
        return self.dt >= self.maxTime

    def scenario_failure(self, observation):
        """Override to identify failures reported by scenario observations."""
        return None

    def terminal_reward(self, observation, reason):
        """Optional reward adjustment for a terminal transition."""
        return 0.0

    def termination_reason(self, observation):
        failure = self.scenario_failure(observation)
        if self._episode_end == "simulation_failure":
            return self._episode_end
        if failure:
            return failure
        if self.dt >= self.maxTime:
            return "time_limit"
        if self._episode_end:
            return self._episode_end
        if self.is_done(observation):
            return "scenario_finished"
        return None

    def reset(self):
        self.close()
        try:
            return self._reset_episode()
        except BaseException:
            self.loop.run_until_complete(self._recover_failure())
            raise

    def _reset_episode(self):
        # Synchronously start a new simulation
        self.dt = 0.0
        self._episode_end = None
        self._simulation_error = None
        self._last_observation = None
        self.loop.run_until_complete(self.start_simulation())  # Run the async function
        self.loop.run_until_complete(self.read_simulation_time())
        print("Simulation started")

        # Get the initial observation
        observation = self.loop.run_until_complete(self._bounded(self.get_observation()))  # Run the async function
        if self._episode_end:
            raise RuntimeError("TestRun ended during reset: " + self._episode_end)
        self._last_observation = observation.copy()
        print("Observation: ", observation)
        return observation

    def step(self, action):
        if self.state != SimulatorState.PAUSED:
            raise RuntimeError("Simulator is not ready; call reset() before step()")
        try:
            return self._step_episode(action)
        except BaseException:
            self.loop.run_until_complete(self._recover_failure())
            raise

    def _step_episode(self, action):
        # Apply action to CarMaker and get new observation synchronously
        previous_time = self.dt
        if not self.loop.run_until_complete(self._bounded(self._check_simulation_end())):
            self.loop.run_until_complete(self._bounded(self.set_action(action)))
            self.loop.run_until_complete(self.advance_simulation())
        observation_valid = True
        try:
            observation = self.loop.run_until_complete(self._bounded(self.get_observation()))
        except Exception:
            if not self._episode_end or self._last_observation is None:
                raise
            observation = self._last_observation.copy()
            observation_valid = False
        self._last_observation = observation.copy()
        if self.dt % 20 == 0:
            print("Observation: ", observation)

        # Compute reward
        reward = self.compute_reward(observation)

        reason = self.termination_reason(observation)
        done = reason is not None
        terminal_reward = 0.0
        if done and observation_valid:
            terminal_reward = self.terminal_reward(observation, reason)
            reward += terminal_reward

        # Additional info (optional)
        info = {"simulation_time": self.dt, "simulation_dt": self.dt - previous_time}
        if done:
            info.update({"termination_reason": reason,
                         "TimeLimit.truncated": reason == "time_limit"})
            if self._simulation_error is not None:
                info["simulation_error"] = self._simulation_error
            info["terminal_observation_valid"] = observation_valid
            info["terminal_reward"] = terminal_reward

        # Automatically stop the simulation if done
        if done:
            print("Reward: ", reward)
            self.close()

        return observation, reward, done, info

    def render(self, mode='human'):
        # Optional: Implement if you want to visualize the simulation
        pass

    def close(self):
        """Close the current simulator session; repeated calls are harmless."""
        if self._simulator_active:
            self.loop.run_until_complete(self.stop_simulation())

