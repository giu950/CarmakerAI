"""Evaluate one episode using a trained PPO policy."""
import argparse

from carmakerv1_env import CarMakerEnvV1
from parametrization_testrun import cmapi
from stable_baselines3 import PPO


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/ppo_carmaker")
    args = parser.parse_args()
    model = PPO.load(args.model)
    env = CarMakerEnvV1()
    observation = env.reset()
    for _ in range(int(env.maxTime / env.time_step) + 1):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, _ = env.step(action)
        if done:
            print("Episode finished. Final reward:", reward)
            break
    else:
        env.close()


if __name__ == "__main__":
    cmapi.Task.run_main_task(main())
