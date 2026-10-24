"""Throttle-control environment for the Bernina Pass example."""
import numpy as np
from gym import spaces

from carmaker_env import CarMakerEnv
from cmapi import DVADuration


class BerninaEnv(CarMakerEnv):
    def __init__(self):
        super().__init__()
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

    def compute_reward(self, observation):
        distance, time, velocity = observation
        return distance + velocity - 0.1 * time

    def is_done(self, observation):
        return super().is_done(observation) or (
            round(observation[2], 1) == 0.0 and self.dt > 5.0
        )
