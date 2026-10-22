# CarmakerAI

A minimal PPO reinforcement learning experiment for IPG CarMaker's Bernina Pass scenario. The agent controls **throttle only**; CarMaker's driver handles steering. This is an experimental baseline, not a complete autonomous driving system.

## Contents

```text
python/
  carmakerv1_env.py          Gym environment and reward
  parametrization_testrun.py CarMaker API and scenario configuration
  train.py                  Train and save a PPO policy
  evaluate.py               Evaluate one episode
carmaker/Data/
  Config/                   Minimal project configuration
  TestRun/berninaPass_v1     Bernina scenario
requirements.txt
```

The course project, duplicate scenarios, archives, trained models, simulation outputs, caches, and compiled binaries are omitted.

## Setup

The original project targets **Windows, CarMaker 11.0, and Python 3.6**. A licensed CarMaker installation and its Python API are required. The test run references the installation's `Examples/Demo_Audi_R8` vehicle and `Examples/Measured/Scenario/CHE_BerninaPass.bin` road; these assets are not bundled here.

Create the legacy environment with Conda, then install dependencies:

```powershell
conda create -n carmaker-ai python=3.6 pip=21.3.1 -y
conda activate carmaker-ai
python -m pip install setuptools==59.6.0 wheel==0.37.1
python -m pip install -r requirements.txt
```

The default API directory is `C:/IPG/carmaker/win64-11.0/Python/python3.6`. Override it if needed:

```powershell
$env:CARMAKER_PYTHON_PATH = "C:/IPG/carmaker/win64-11.0/Python/python3.6"
```

Open `carmaker/` as a project in CarMaker 11.0 and check that `berninaPass_v1` runs with your installation. If the minimal project needs local initialization, use CarMaker's project setup/update workflow. Alternatively, set `CARMAKER_PROJECT` to an existing initialized BerninaAI project containing this test run. Set `CARMAKER_TESTRUN` to select another test run.

## Train and evaluate

Run from the repository root:

```powershell
python python/train.py --timesteps 2048 --model models/ppo_carmaker
python python/evaluate.py --model models/ppo_carmaker
```

Models and simulation results are ignored by Git. PPO collects full rollout batches, so the actual number of training steps can exceed the requested minimum.

## Environment

- Action: one throttle value between 0.02 and 1.0.
- Observation: road distance, simulation time, and vehicle speed.
- Reward: `distance + speed - 0.1 * time`.
- Episode ends at the environment's time limit or when the vehicle stops after five seconds. CarMaker's test run also has its own time limit (180 seconds).

The legacy Gym API is retained to match Stable-Baselines3 1.3.0. The environment uses nested asynchronous simulation calls. Its shutdown waits for CarMaker to report a finished simulation; early closure may therefore wait until the test run finishes. Simulation timing, episode handling, and reward quality need validation in CarMaker before serious training.

## Validation

Python source syntax was checked with Python 3.6 grammar. End-to-end simulation and training have not been validated for this cleaned version; they require the licensed CarMaker installation and the legacy dependency environment. Modern Python versions are not the target for this dependency stack.
