"""Train a PPO policy using the legacy Gym environment."""
import argparse
from pathlib import Path

from bernina_env import BerninaEnv
from parametrization_testrun import cmapi, DEFAULT_TERMINATION_TIME, termination_time_seconds
from stable_baselines3 import PPO


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--model", default="models/ppo_carmaker")
    parser.add_argument("--termination-time", type=termination_time_seconds,
                        default=DEFAULT_TERMINATION_TIME,
                        help="Episode and TestRun time limit in seconds (default: 180)")
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    env = BerninaEnv(termination_time=args.termination_time)
    try:
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=args.timesteps)
        model.save(args.model)
    finally:
        env.close()


if __name__ == "__main__":
    cmapi.Task.run_main_task(main())
