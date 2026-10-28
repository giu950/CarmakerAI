"""Environment regression checks using simulated CarMaker I/O."""
import asyncio
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from carmaker_env import CarMakerEnv, SimulatorState
from bernina_env import BerninaEnv
from cmapi import DVADuration
import cmapi
import train


async def no_simulation_errors():
    pass


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

    def test_reward_counts_only_new_progress(self):
        env = BerninaEnv()
        self.assertEqual(env.compute_reward([1000.0, 10.0, 12.0]), 0.0)
        self.assertAlmostEqual(env.compute_reward([1002.0, 10.02, 12.0]), 1.998)
        self.assertAlmostEqual(env.compute_reward([1002.0, 10.04, 12.0]), -0.002)

    def test_reward_penalizes_reverse_progress(self):
        env = BerninaEnv()
        env.compute_reward([1000.0, 10.0, 12.0])
        self.assertAlmostEqual(env.compute_reward([999.0, 10.02, 12.0]), -1.002)

    def test_stopping_adds_penalty_after_startup_grace(self):
        env = BerninaEnv()
        env.compute_reward([1000.0, 5.0, 0.0])
        self.assertAlmostEqual(env.compute_reward([1000.0, 5.02, 0.0]), -0.022)
        env.compute_reward([1000.0, 0.0, 0.0])
        self.assertAlmostEqual(env.compute_reward([1000.0, 0.02, 0.0]), -0.002)

    def test_reward_time_penalties_use_actual_elapsed_time(self):
        env = BerninaEnv()
        env.compute_reward([1000.0, 10.0, 12.0])
        reward = env.compute_reward([1001.0, 10.03, 12.0])
        reward += env.compute_reward([1002.0, 10.10, 12.0])
        self.assertAlmostEqual(reward, 2.0 - 0.1 * 0.1)

    def test_reset_reinitializes_reward_baseline(self):
        env = self.make_env()
        env.reset()
        env.compute_reward([1000.0, 10.0, 12.0])
        observation = env.reset()
        self.assertEqual(observation[0], 100.0)
        self.assertAlmostEqual(env.compute_reward([101.0, 0.02, 12.0]), 0.998)

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
            env.state = SimulatorState.PAUSED
            env.stopped = False
            env.state = SimulatorState.PAUSED

        async def stop():
            env.stopped = True
            env.state = SimulatorState.CLOSED
            env._simulator_active = False

        env.stopped = False
        env._simulator_active = True
        env.state = SimulatorState.PAUSED
        env.start_simulation = start
        env.stop_simulation = stop
        env.simcontrol = SimpleNamespace(
            raise_sim_errors=no_simulation_errors,
            get_simstate=Mock(return_value=SimpleNamespace(
                get_condition_simstate=Mock(return_value=cmapi.ConditionSimState.running))),
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
            raise_sim_errors=no_simulation_errors,
            get_simstate=Mock(return_value=SimpleNamespace(
                get_condition_simstate=Mock(return_value=cmapi.ConditionSimState.running))),
            stop_sim=stop, stop_and_disconnect=disconnect,
            create_simstate_condition=Mock(return_value=SimpleNamespace(wait=wait)),
        )
        env._simulator_active = True
        env.state = SimulatorState.PAUSED
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

    def test_step_requires_reset_after_failure_or_close(self):
        env = BerninaEnv()
        for state in (SimulatorState.CLOSED, SimulatorState.FAILED):
            env.state = state
            with self.assertRaisesRegex(RuntimeError, "reset"):
                env.step([0.5])

    def test_shutdown_finished_wait_has_timeout(self):
        env, calls = self.make_cleanup_env()
        env.sync_timeout = 0.01

        async def hang():
            await asyncio.Future()

        env.simcontrol.create_simstate_condition.return_value.wait = hang
        with self.assertRaises(asyncio.TimeoutError):
            env.close()
        self.assertEqual(calls, ["stop", "disconnect"])
        self.assertEqual(env.state, SimulatorState.FAILED)
        self.assertIsNone(env.simcontrol)

    def test_disconnect_timeout_uses_fallback_and_discards_controller(self):
        env, calls = self.make_cleanup_env()
        env.sync_timeout = 0.01

        async def hang():
            await asyncio.Future()

        async def fallback():
            calls.append("fallback")

        env.simcontrol.stop_and_disconnect = hang
        env.simcontrol.disconnect = fallback
        with self.assertRaises(asyncio.TimeoutError):
            env.close()
        self.assertEqual(calls, ["stop", "finished", "fallback"])
        self.assertEqual(env.state, SimulatorState.FAILED)
        self.assertIsNone(env.simcontrol)
        env.close()

    def test_step_error_survives_pause_and_shutdown_errors(self):
        env = self.make_env()

        async def fail_wait():
            raise RuntimeError("original step error")

        async def fail_cleanup():
            raise ValueError("cleanup error")

        env.simcontrol.create_quantity_condition.return_value.wait = fail_wait
        env.simcontrol.pause_sim.side_effect = ValueError("pause error")
        env.stop_simulation = fail_cleanup
        with self.assertLogs("carmaker_env", level="WARNING"):
            with self.assertRaisesRegex(RuntimeError, "original step error"):
                env.step([0.5])
        self.assertEqual(env.state, SimulatorState.FAILED)
        self.assertIsNone(env.simcontrol)
        self.assertFalse(env._simulator_active)

    def test_missing_observation_disconnects_and_marks_failed(self):
        env, calls = self.make_cleanup_env()

        async def no_op(*args):
            pass

        async def missing():
            raise KeyError("quantity unavailable")

        env.set_action = no_op
        env.advance_simulation = no_op
        env.get_observation = missing
        with self.assertRaisesRegex(KeyError, "quantity unavailable"):
            env.step([0.5])
        self.assertEqual(calls, ["stop", "finished", "disconnect"])
        self.assertEqual(env.state, SimulatorState.FAILED)

    def test_action_timeout_disconnects_and_marks_failed(self):
        env, calls = self.make_cleanup_env()
        env.sync_timeout = 0.01

        async def hang(action):
            await asyncio.Future()

        env.set_action = hang
        with self.assertRaises(asyncio.TimeoutError):
            env.step([0.5])
        self.assertEqual(calls, ["stop", "finished", "disconnect"])
        self.assertEqual(env.state, SimulatorState.FAILED)
        self.assertIsNone(env.simcontrol)

    def test_partial_startup_timeout_then_reset_uses_new_controller(self):
        env = BerninaEnv()
        env.sync_timeout = 0.01
        controllers = []
        disconnected = []

        def factory():
            index = len(controllers)

            async def no_op(*args):
                pass

            async def connect():
                if index == 0:
                    await asyncio.Future()

            async def disconnect():
                disconnected.append(index)

            async def read(*quantities):
                values = {"Time": 0.0, "SC.TAccel": 0.0,
                          "Car.Road.sRoad": 100.0, "Car.v": 12.0}
                return [values[name] for name in quantities]

            controller = SimpleNamespace(
                raise_sim_errors=no_simulation_errors,
                get_simstate=Mock(return_value=SimpleNamespace(
                    get_condition_simstate=Mock(return_value=cmapi.ConditionSimState.running))),
                set_variation=Mock(), set_master=no_op,
                start_and_connect=connect, start_sim=no_op,
                pause_sim=Mock(), stop_sim=no_op, stop_and_disconnect=disconnect,
                create_simstate_condition=Mock(
                    return_value=SimpleNamespace(wait=no_op)),
                simio=SimpleNamespace(dva_read_async=read),
            )
            controllers.append(controller)
            return controller

        async def variations(**kwargs):
            return [Mock()]

        with patch("carmaker_env.make_variations", variations), \
                patch("carmaker_env.cmapi.SimControlInteractive", side_effect=factory), \
                patch("carmaker_env.cmapi.CarMaker"):
            with self.assertRaises(asyncio.TimeoutError):
                env.reset()
            self.assertEqual(env.state, SimulatorState.FAILED)
            self.assertIsNone(env.simcontrol)
            self.assertEqual(disconnected, [0])
            env.reset()
            self.assertEqual(env.state, SimulatorState.PAUSED)
            self.assertIs(env.simcontrol, controllers[1])
            self.assertIsNot(controllers[0], controllers[1])
            env.close()
        self.assertEqual(disconnected, [0, 1])
        self.assertEqual(env.state, SimulatorState.CLOSED)

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
            env.state = SimulatorState.PAUSED
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
        self.assertAlmostEqual(reward, -0.002, places=5)
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

    def end_during_step(self, failed=False, at_limit=False):
        env = self.make_env()
        env.sync_timeout = 0.1
        cancelled = []

        async def finish_without_time_progress():
            env.simcontrol.get_simstate().get_condition_simstate.return_value = (
                cmapi.ConditionSimState.finished)
            if at_limit:
                env.simulation_time = env.maxTime
            if failed:
                async def report_error():
                    raise cmapi.SimulationError("TestRun aborted")
                env.simcontrol.raise_sim_errors = report_error
            try:
                await asyncio.Future()
            finally:
                cancelled.append(True)

        env.simcontrol.create_quantity_condition.return_value.wait = finish_without_time_progress
        result = env.step([0.5])
        self.assertTrue(cancelled)
        self.assertTrue(env.stopped)
        env.simcontrol.pause_sim.assert_not_called()
        return result

    def test_finished_testrun_ends_without_clock_progress(self):
        _, _, done, info = self.end_during_step()
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "simulation_finished")
        self.assertFalse(info["TimeLimit.truncated"])

    def test_simulation_error_is_failure_without_clock_progress(self):
        _, _, done, info = self.end_during_step(failed=True)
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "simulation_failure")
        self.assertIn("TestRun aborted", info["simulation_error"])
        self.assertFalse(info["TimeLimit.truncated"])

    def test_failure_at_time_limit_is_not_truncation(self):
        _, _, _, info = self.end_during_step(failed=True, at_limit=True)
        self.assertEqual(info["termination_reason"], "simulation_failure")
        self.assertFalse(info["TimeLimit.truncated"])

    def test_finished_at_time_limit_is_truncation(self):
        _, _, _, info = self.end_during_step(at_limit=True)
        self.assertEqual(info["termination_reason"], "time_limit")
        self.assertTrue(info["TimeLimit.truncated"])

    def test_stopped_vehicle_is_scenario_failure(self):
        env = self.make_env(speed=0.0, simulation_time=6.0)
        _, _, done, info = env.step([0.5])
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "vehicle_stopped")
        self.assertFalse(info["TimeLimit.truncated"])

    def test_end_between_steps_does_not_apply_action(self):
        env = self.make_env()
        env.reset()
        env.simcontrol.get_simstate().get_condition_simstate.return_value = (
            cmapi.ConditionSimState.finished)
        _, _, done, info = env.step([0.5])
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "simulation_finished")
        env.simcontrol.simio.dva_write_absolute_value.assert_not_called()
        env.simcontrol.resume_sim.assert_not_called()

    def test_finished_session_with_unavailable_observation_uses_last_observation(self):
        env = self.make_env()
        initial = env.reset()
        env.simcontrol.get_simstate().get_condition_simstate.return_value = (
            cmapi.ConditionSimState.finished)

        async def unavailable():
            raise KeyError("quantities unavailable after finish")

        env.get_observation = unavailable
        observation, _, done, info = env.step([0.5])
        np.testing.assert_array_equal(observation, initial)
        self.assertTrue(done)
        self.assertFalse(info["terminal_observation_valid"])

    def test_finished_session_disconnects_without_stop_or_finished_wait(self):
        env, calls = self.make_cleanup_env()
        env.simcontrol.get_simstate().get_condition_simstate.return_value = (
            cmapi.ConditionSimState.finished)
        self.assertTrue(self.loop.run_until_complete(env._check_simulation_end()))
        env.close()
        self.assertEqual(calls, ["disconnect"])

    def test_finish_while_waiting_for_pause_acknowledgement(self):
        env = self.make_env()
        env.simcontrol.get_simstate().get_condition_simstate.return_value = (
            cmapi.ConditionSimState.finished)

        async def unavailable(*quantities):
            self.fail("Finished TestRun must not wait for pause acknowledgement")

        env.simcontrol.simio.dva_read_async = unavailable
        self.loop.run_until_complete(env.pause_simulation())
        self.assertEqual(env._episode_end, "simulation_finished")


if __name__ == "__main__":
    unittest.main()
