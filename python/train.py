"""Train a PPO policy using the legacy Gym environment."""
import argparse
from pathlib import Path

from bernina_env import BerninaEnv
from parametrization_testrun import cmapi
from stable_baselines3 import PPO


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--model", default="models/ppo_carmaker")
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    env = BerninaEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model)
    env.close()


if __name__ == "__main__":
    cmapi.Task.run_main_task(main())
