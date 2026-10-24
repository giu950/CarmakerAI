"""Reusable Gym lifecycle for CarMaker environments."""
from abc import ABC, abstractmethod

import gym
from parametrization_testrun import cmapi, make_variations
import asyncio
import nest_asyncio  # Import nest_asyncio

class CarMakerEnv(gym.Env, ABC):
    def __init__(self):
        super().__init__()

        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()

        # Keep the original simulation timing and episode behavior.
        self.dt = 0.0
        self.time_step = 0.02
        self.maxTime = 890
        self.loop = asyncio.get_event_loop()  # Create an event loop
        print("Init success")

    async def start_simulation(self):
        variations = await make_variations()

        cmapi.logger.info(f"Executing with self.simcontrol...")
        self.variation = variations[0]

        self.simcontrol = cmapi.SimControlInteractive()
        print("OK self.simcontrolInteractive")
        self.simcontrol.set_variation(self.variation)
        print("OK set_variation")
        master = cmapi.CarMaker()
        await self.simcontrol.set_master(master)
        print("OK set_master")
        await self.simcontrol.start_and_connect()
        print("OK start_and_connect")
        await self.simcontrol.start_sim()
        print("OK start_sim")

    async def stop_simulation(self):
        print("Stopping simulation")
        await self.simcontrol.create_simstate_condition(cmapi.ConditionSimState.finished).wait()
        await self.simcontrol.stop_and_disconnect()
        print("Simulation stopped")

    @abstractmethod
    async def set_action(self, action):
        """Apply an action and wait for the simulation step to complete."""
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
        return self.dt > self.maxTime

    def reset(self):
        # Synchronously start a new simulation
        self.dt = 0.0
        self.loop.run_until_complete(self.start_simulation())  # Run the async function
        print("Simulation started")

        # Get the initial observation
        observation = self.loop.run_until_complete(self.get_observation())  # Run the async function
        print("Observation: ", observation)
        return observation

    def step(self, action):
        # Apply action to CarMaker and get new observation synchronously
        self.loop.run_until_complete(self.set_action(action))  # Run the async function
        self.dt += self.time_step
        observation = self.loop.run_until_complete(self.get_observation())  # Run the async function
        if self.dt % 20 == 0:
            print("Observation: ", observation)

        # Compute reward
        reward = self.compute_reward(observation)

        done = self.is_done(observation)

        # Additional info (optional)
        info = {}

        # Automatically stop the simulation if done
        if done:
            print("Reward: ", reward)
            self.close()

        return observation, reward, done, info

    def render(self, mode='human'):
        # Optional: Implement if you want to visualize the simulation
        pass

    def close(self):
        # Synchronously stop the simulation
        self.loop.run_until_complete(self.stop_simulation())  # Run the async function

