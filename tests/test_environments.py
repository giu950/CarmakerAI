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

    def make_env(self, speed=12.0, simulation_time=0.0, overshoot=0.0):
        env = BerninaEnv()

        env.simulation_time = simulation_time
        env.paused = True
        env.events = []

        async def read(*quantities):
            return tuple({"Car.Road.sRoad": 100.0, "Time": env.simulation_time,
                          "Car.v": speed, "SC.TAccel": 0.0 if env.paused else 1.0
                          }[quantity] for quantity in quantities)

        async def wait():
            self.assertFalse(env.paused)
            predicate, _ = env.simcontrol.create_quantity_condition.call_args[0]
            self.assertFalse(predicate(env.simulation_time))
            env.simulation_time += env.time_step + overshoot
            self.assertTrue(predicate(env.simulation_time))
            env.events.append("advance")

        def pause():
            env.paused = True
            env.events.append("pause")

        def resume():
            env.paused = False
            env.events.append("resume")

        def write(*args):
            self.assertTrue(env.paused)
            env.events.append("action")

        async def start():
            env.simulation_time = simulation_time
            env.paused = True

        async def stop():
            env.stopped = True

        env.stopped = False
        env.start_simulation = start
        env.stop_simulation = stop
        env.simcontrol = SimpleNamespace(
            simio=SimpleNamespace(
                dva_read_async=read, dva_write_absolute_value=Mock(side_effect=write),
                dva_release=Mock(),
            ),
            pause_sim=Mock(side_effect=pause),
            resume_sim=Mock(side_effect=resume),
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
        self.assertEqual(info, {"simulation_time": 0.02, "simulation_dt": 0.02})
        self.assertEqual(env.events, ["action", "resume", "advance", "pause"])
        self.assertTrue(env.paused)
        self.assertFalse(env.stopped)
        env.simcontrol.simio.dva_write_absolute_value.assert_called_once_with(
            "DM.Gas", 0.5, DVADuration.permanent
        )
        condition, quantity = env.simcontrol.create_quantity_condition.call_args[0]
        self.assertEqual(quantity, "Time")
        self.assertFalse(condition(0.0))
        self.assertTrue(condition(0.02))
        env.simcontrol.simio.dva_release.assert_not_called()

    def test_clock_tracks_startup_offset_and_communication_overshoot(self):
        env = self.make_env(simulation_time=1.25, overshoot=0.007)
        observation = env.reset()
        self.assertAlmostEqual(env.dt, 1.25)
        self.assertAlmostEqual(observation[1], 1.25)
        for _ in range(3):
            previous_time = env.dt
            observation, _, done, info = env.step([0.5])
            self.assertFalse(done)
            self.assertAlmostEqual(env.dt, env.simulation_time)
            self.assertAlmostEqual(observation[1], env.dt, places=6)
            self.assertAlmostEqual(info["simulation_dt"], 0.027)
            predicate, _ = env.simcontrol.create_quantity_condition.call_args[0]
            self.assertFalse(predicate(previous_time))
            self.assertTrue(predicate(previous_time + 0.02))

    def test_pause_after_wait_failure(self):
        env = self.make_env()

        async def fail():
            raise RuntimeError("Simulation disconnected")

        env.simcontrol.create_quantity_condition.return_value.wait = fail
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            env.step([0.5])
        self.assertTrue(env.paused)

    def test_pause_waits_for_simulator_acknowledgement(self):
        env = self.make_env()
        readings = iter([1.0, 1.0, 0.0])
        calls = []

        async def read(quantity):
            calls.append(quantity)
            return (next(readings),)

        env.simcontrol.simio.dva_read_async = read
        self.loop.run_until_complete(env.pause_simulation())
        self.assertEqual(calls, ["SC.TAccel"] * 3)

    def test_stopped_car_ends_only_after_five_seconds(self):
        env = self.make_env(speed=0.0)
        observation = np.array([100.0, 0.0, 0.0])
        env.dt = env.simulation_time = 5.0
        self.assertFalse(env.is_done(observation))
        _, _, done, _ = env.step([0.5])
        self.assertTrue(done)
        self.assertTrue(env.stopped)

    def test_time_limit_ends_moving_car_episode(self):
        env = self.make_env()
        env.dt = env.simulation_time = env.maxTime
        _, _, done, _ = env.step([0.5])
        self.assertTrue(done)
        self.assertTrue(env.stopped)


if __name__ == "__main__":
    unittest.main()
