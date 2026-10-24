"""Environment regression checks using simulated CarMaker I/O."""
import asyncio
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from carmaker_env import CarMakerEnv
from bernina_env import BerninaEnv
from cmapi import DVADuration


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(self.loop)
        self.addCleanup(asyncio.set_event_loop, None)
        self.addCleanup(self.loop.close)

    def test_base_requires_scenario_methods(self):
        with self.assertRaises(TypeError):
            CarMakerEnv()

    def make_env(self, speed=12.0):
        env = BerninaEnv()

        async def read(quantity):
            return ({"Car.Road.sRoad": 100.0, "Time": env.dt,
                     "Car.v": speed, "DM.Gas": 0.5}[quantity],)

        async def wait():
            pass

        async def start():
            pass

        async def stop():
            env.stopped = True

        env.stopped = False
        env.start_simulation = start
        env.stop_simulation = stop
        env.simcontrol = SimpleNamespace(
            simio=SimpleNamespace(
                dva_read_async=read, dva_write_absolute_value=Mock(),
                dva_release=Mock(),
            ),
            create_quantity_condition=Mock(
                return_value=SimpleNamespace(wait=wait)
            ),
        )
        return env

    def test_reset_and_step_preserve_bernina_behavior(self):
        env = self.make_env()
        env.dt = 10.0
        observation = env.reset()
        self.assertEqual(env.dt, 0.0)
        self.assertTrue(env.observation_space.contains(observation))
        observation, reward, done, info = env.step(np.array([0.5]))
        self.assertAlmostEqual(env.dt, 0.02)
        np.testing.assert_allclose(observation, [100.0, 0.02, 12.0])
        self.assertAlmostEqual(reward, 111.998, places=5)
        self.assertFalse(done)
        self.assertEqual(info, {})
        self.assertFalse(env.stopped)
        env.simcontrol.simio.dva_write_absolute_value.assert_called_once_with(
            "DM.Gas", 0.5, DVADuration.permanent
        )
        condition, quantity = env.simcontrol.create_quantity_condition.call_args[0]
        self.assertEqual(quantity, "Time")
        self.assertFalse(condition(env.dt))
        self.assertTrue(condition(env.dt + env.time_step))
        self.assertEqual(env.simcontrol.simio.dva_release.call_count, 2)

    def test_stopped_car_ends_only_after_five_seconds(self):
        env = self.make_env(speed=0.0)
        observation = np.array([100.0, 0.0, 0.0])
        env.dt = 5.0
        self.assertFalse(env.is_done(observation))
        _, _, done, _ = env.step([0.5])
        self.assertTrue(done)
        self.assertTrue(env.stopped)

    def test_time_limit_ends_moving_car_episode(self):
        env = self.make_env()
        env.dt = env.maxTime
        _, _, done, _ = env.step([0.5])
        self.assertTrue(done)
        self.assertTrue(env.stopped)


if __name__ == "__main__":
    unittest.main()
