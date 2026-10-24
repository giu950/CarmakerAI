"""Environment regression checks using simulated CarMaker I/O."""
import asyncio
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from carmaker_env import CarMakerEnv
from bernina_env import BerninaEnv
from cmapi import DVADuration
import train


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(self.loop)
        self.addCleanup(asyncio.set_event_loop, None)
        self.addCleanup(self.loop.close)

    def test_base_requires_scenario_methods(self):
        with self.assertRaises(TypeError):
            CarMakerEnv()

    def test_custom_termination_time(self):
        env = BerninaEnv(termination_time=240)
        env.dt = 180.0
        self.assertFalse(env.is_done([100.0, 180.0, 12.0]))
        env.dt = 240.0
        self.assertTrue(env.is_done([100.0, 240.0, 12.0]))

    def test_invalid_termination_times(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BerninaEnv(termination_time=value)

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
            env._simulator_active = True
            env.stopped = False

        async def stop():
            env.stopped = True
            env._simulator_active = False

        env.stopped = False
        env._simulator_active = True
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

    def make_cleanup_env(self, stop_error=None):
        env = BerninaEnv()
        calls = []

        async def stop():
            calls.append("stop")
            if stop_error:
                raise stop_error

        async def wait():
            calls.append("finished")

        async def disconnect():
            calls.append("disconnect")

        env.simcontrol = SimpleNamespace(
            stop_sim=stop, stop_and_disconnect=disconnect,
            create_simstate_condition=Mock(return_value=SimpleNamespace(wait=wait)),
        )
        env._simulator_active = True
        env._simulation_started = True
        return env, calls

    def test_close_is_idempotent(self):
        env, calls = self.make_cleanup_env()
        env.close()
        env.close()
        self.assertEqual(calls, ["stop", "finished", "disconnect"])
        self.assertIsNone(env.simcontrol)

    def test_disconnect_runs_even_when_stop_fails(self):
        env, calls = self.make_cleanup_env(RuntimeError("stop failed"))
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            env.close()
        env.close()
        self.assertEqual(calls, ["stop", "disconnect"])

    def test_close_before_start_is_harmless(self):
        env = BerninaEnv()
        env.close()
        self.assertFalse(env._simulator_active)

    def test_episode_end_followed_by_close_stops_only_once(self):
        env, calls = self.make_cleanup_env()

        async def action(value):
            pass

        async def advance():
            env.dt = env.maxTime

        async def observation():
            return np.array([100.0, env.dt, 12.0], dtype=np.float32)

        env.set_action = action
        env.advance_simulation = advance
        env.get_observation = observation
        _, _, done, _ = env.step([0.5])
        self.assertTrue(done)
        env.close()
        self.assertEqual(calls, ["stop", "finished", "disconnect"])

    def test_training_errors_always_close_environment(self):
        for stage in ("construct", "learn", "save"):
            with self.subTest(stage=stage):
                env = Mock()
                model = Mock()
                failure = RuntimeError("training failed")
                if stage != "construct":
                    getattr(model, stage).side_effect = failure
                with patch.object(sys, "argv", ["train.py"]), \
                        patch.object(train, "BerninaEnv", return_value=env), \
                        patch.object(train, "Path"), \
                        patch.object(train, "PPO", return_value=model) as ppo:
                    if stage == "construct":
                        ppo.side_effect = failure
                    with self.assertRaisesRegex(RuntimeError, "training failed"):
                        self.loop.run_until_complete(train.main())
                env.close.assert_called_once_with()

    def test_partial_startup_disconnects_without_stopping_simulation(self):
        env, calls = self.make_cleanup_env()
        env._simulation_started = False
        env.close()
        self.assertEqual(calls, ["disconnect"])

    def test_reset_failure_cleans_up_connection(self):
        env, calls = self.make_cleanup_env()
        env.close()
        calls.clear()

        async def start():
            replacement, _ = self.make_cleanup_env()
            env.simcontrol = replacement.simcontrol
            env._simulator_active = True
            env._simulation_started = True
            raise RuntimeError("startup failed")

        env.start_simulation = start
        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            env.reset()
        self.assertFalse(env._simulator_active)
        self.assertIsNone(env.simcontrol)

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
        env.dt = env.simulation_time = env.maxTime - env.time_step
        _, _, done, _ = env.step([0.5])
        self.assertEqual(env.dt, env.maxTime)
        self.assertTrue(done)
        self.assertTrue(env.stopped)

    def test_episode_limit_matches_bundled_testrun(self):
        testrun = (pathlib.Path(__file__).resolve().parents[1]
                   / "carmaker" / "Data" / "TestRun" / "berninaPass")
        parameters = dict(
            line.split(" = ", 1) for line in testrun.read_text().splitlines()
            if " = " in line
        )
        self.assertEqual(BerninaEnv().maxTime,
                         float(parameters["DrivMan.0.TimeLimit"]))

    def test_final_step_does_not_wait_past_episode_limit(self):
        env = self.make_env()
        env.dt = env.simulation_time = env.maxTime - 0.005
        env.step([0.5])
        predicate, _ = env.simcontrol.create_quantity_condition.call_args[0]
        self.assertFalse(predicate(env.maxTime - 0.001))
        self.assertTrue(predicate(env.maxTime))


if __name__ == "__main__":
    unittest.main()
