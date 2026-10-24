"""Evaluate one episode using a trained PPO policy."""
import argparse

from bernina_env import BerninaEnv
from parametrization_testrun import cmapi, DEFAULT_TERMINATION_TIME, termination_time_seconds
from stable_baselines3 import PPO


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/ppo_carmaker")
    parser.add_argument("--termination-time", type=termination_time_seconds,
                        default=DEFAULT_TERMINATION_TIME,
                        help="Episode and TestRun time limit in seconds (default: 180)")
    args = parser.parse_args()
    model = PPO.load(args.model)
    env = BerninaEnv(termination_time=args.termination_time)
    try:
        observation = env.reset()
        for _ in range(int(env.maxTime / env.time_step) + 1):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, _ = env.step(action)
            if done:
                print("Episode finished. Final reward:", reward)
                break
    finally:
        env.close()


if __name__ == "__main__":
    cmapi.Task.run_main_task(main())
