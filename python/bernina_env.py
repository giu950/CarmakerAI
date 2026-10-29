"""Throttle-control environment for the Bernina Pass example."""
import math
import numpy as np
from gym import spaces

from carmaker_env import CarMakerEnv
from parametrization_testrun import DEFAULT_TERMINATION_TIME
from cmapi import DVADuration


class BerninaEnv(CarMakerEnv):
    def __init__(self, termination_time=DEFAULT_TERMINATION_TIME,
                 terminal_failure_penalties=None):
        if terminal_failure_penalties is None:
            terminal_failure_penalties = {
                "collision": 100.0, "road_departure": 100.0,
                "vehicle_stopped": 100.0,
            }
        self.terminal_failure_penalties = {
            reason: float(penalty)
            for reason, penalty in terminal_failure_penalties.items()
        }
        if any(not math.isfinite(penalty) or penalty < 0.0
               for penalty in self.terminal_failure_penalties.values()):
            raise ValueError("terminal failure penalties must be finite and nonnegative")
        super().__init__(termination_time=termination_time)
        self.time_penalty_rate = 0.1
        self.stopped_penalty_rate = 1.0
        self._previous_distance = None
        self._previous_time = None
        # The CarMaker driver handles steering.
        self.action_space = spaces.Box(
            low=0.02, high=1.0, shape=(1,), dtype=np.float32
        )
        # Road position and velocity are signed, with no scenario-independent
        # finite limits. Simulation time remains nonnegative and may overshoot
        # the episode limit during an APO step.
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, -np.inf], dtype=np.float32),
            high=np.full(3, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    async def set_action(self, action):
        """Apply one finite throttle value within the declared action bounds."""
        try:
            action = np.asarray(action)
        except (TypeError, ValueError) as exc:
            raise ValueError("action must be a numeric array with shape (1,)") from exc
        if action.shape != self.action_space.shape:
            raise ValueError("action must have shape (1,)")
        if action.dtype.kind not in "fiu" or not np.all(np.isfinite(action)):
            raise ValueError("action must contain a finite real number")
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            raise ValueError("action must be within the throttle bounds [0.02, 1.0]")
        self.simcontrol.simio.dva_write_absolute_value(
            "DM.Gas", float(action[0]),
            DVADuration.permanent,
        )

    async def get_observation(self):
        distance, time, velocity = await self.simcontrol.simio.dva_read_async(
            "Car.Road.sRoad", "Time", "Car.v"
        )
        return np.array([distance, time, velocity], dtype=np.float32)

    def reset(self):
        self._previous_distance = None
        self._previous_time = None
        observation = super().reset()
        self._previous_distance = float(observation[0])
        self._previous_time = float(observation[1])
        return observation

    def compute_reward(self, observation):
        """Reward new road progress and penalize elapsed time and stopping."""
        distance, time, velocity = map(float, observation)
        if self._previous_distance is None:
            self._previous_distance = distance
            self._previous_time = time
            return 0.0

        progress = distance - self._previous_distance
        elapsed = max(0.0, time - self._previous_time)
        self._previous_distance = distance
        self._previous_time = time

        reward = progress - self.time_penalty_rate * elapsed
        if round(velocity, 1) == 0.0 and time > 5.0:
            reward -= self.stopped_penalty_rate * elapsed
        return reward

    def is_done(self, observation):
        return super().is_done(observation) or bool(self.scenario_failure(observation))

    def terminal_reward(self, observation, reason):
        # Require an identified driving failure, including when CarMaker also
        # reports a generic simulation error. Never infer one from error text.
        failure = self.scenario_failure(observation)
        return -self.terminal_failure_penalties.get(failure, 0.0)

    def scenario_failure(self, observation):
        if round(observation[2], 1) == 0.0 and self.dt > 5.0:
            return "vehicle_stopped"
        return None
