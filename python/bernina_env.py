"""Throttle-control environment for the Bernina Pass example."""
import numpy as np
from gym import spaces

from carmaker_env import CarMakerEnv
from parametrization_testrun import DEFAULT_TERMINATION_TIME
from cmapi import DVADuration


class BerninaEnv(CarMakerEnv):
    def __init__(self, termination_time=DEFAULT_TERMINATION_TIME):
        super().__init__(termination_time=termination_time)
        self.time_penalty_rate = 0.1
        self.stopped_penalty_rate = 1.0
        self._previous_distance = None
        self._previous_time = None
        # The CarMaker driver handles steering.
        self.action_space = spaces.Box(
            low=0.02, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.full(3, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    async def set_action(self, action):
        self.simcontrol.simio.dva_write_absolute_value(
            "DM.Gas", float(np.asarray(action).reshape(-1)[0]),
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
        return super().is_done(observation) or (
            round(observation[2], 1) == 0.0 and self.dt > 5.0
        )
