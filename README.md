# Reinforcement Learning for Autonomous Driving with CarMaker

This repository provides a **minimal working setup for developing autonomous driving agents using Reinforcement Learning (RL) and IPG CarMaker**.

The goal is to offer a starting point for building, training, and evaluating custom RL agents in a CarMaker simulation environment.

To demonstrate the framework and verify the installation, the repository includes a simple example based on the **Bernina Pass scenario**, where a PPO agent controls **only the vehicle's throttle**, while CarMaker's built-in driver handles steering.

The repository can be used in two steps:

1. **Set up and test the framework:** Configure Python and CarMaker, verify the connection, and run the provided throttle-control example.
2. **Develop a custom environment:** Extend the example by defining custom actions, observations, rewards, and simulation scenarios to develop more advanced autonomous driving agents.

## Repository structure

```text
CarmakerAI/
├── python/
│   ├── carmaker_env.py           # Reusable CarMaker Gym base class
?   ??? bernina_env.py            # Bernina actions, observations, and reward
│   ├── parametrization_testrun.py # CarMaker API and scenario configuration
│   ├── test_connection.py        # Check the CarMaker Python API
│   ├── train.py                  # Train and save a PPO policy
│   └── evaluate.py               # Evaluate the trained policy
├── carmaker/
│   └── Data/
│       ├── Config/               # CarMaker project configuration
│       └── TestRun/
│           └── berninaPass        # Bernina Pass scenario
├── requirements.txt
└── README.md
```

The repository contains the minimal files needed for the example. Trained models, simulation outputs, and other generated files are excluded.

# Step 1 — Set up and test the environment

This section explains how to configure the Python environment, connect it to CarMaker, and run the provided Bernina Pass example.

### 1.1 Requirements

The project targets the following configuration:

- Windows
- **IPG CarMaker 11.0** with a valid license
- **Python 3.6**
- Conda (Miniconda or Anaconda)
- Gym and Stable-Baselines3

A working CarMaker installation and its Python API are required. The CarMaker API is provided with CarMaker and is not installed through `requirements.txt`.

### 1.2 Set up the Python environment

We recommend creating a dedicated Conda environment to avoid dependency conflicts.

Clone the repository and navigate to its root directory:

```powershell
git clone https://github.com/giu950/CarmakerAI.git
cd CarmakerAI
```

Create and activate the Conda environment:

```powershell
conda create -n CMai python=3.6 pip=21.3.1 -y
conda activate CMai
```

Install the required dependencies:

```powershell
python -m pip install setuptools==59.6.0 wheel==0.37.1
python -m pip install -r requirements.txt
```

Verify that the correct Python version is installed:

```powershell
python -V
```

The output should indicate Python 3.6.x, for example:

```text
Python 3.6.13 :: Anaconda, Inc.
```

### 1.3 Configure the CarMaker Python API

Once the Python environment is ready, the CarMaker Python API must be added to the Python search path.

For CarMaker 11.0 on Windows, the default installation path is:

```text
C:/IPG/carmaker/win64-11.0/Python/python3.6
```

The path can be added directly to a Python script before importing the API:

```python
import sys
sys.path.append("C:/IPG/carmaker/win64-11.0/Python/python3.6")

import cmapi
```

In this repository, the API path is handled by `python/parametrization_testrun.py`.

If CarMaker is installed in a different directory, configure the `CARMAKER_PYTHON_PATH` environment variable in PowerShell:

```powershell
$env:CARMAKER_PYTHON_PATH = "C:/IPG/carmaker/win64-11.0/Python/python3.6"
```

Replace the path with the location of your CarMaker installation.

**Note:** This environment variable applies to the current PowerShell session. If you open a new terminal, activate the Conda environment again and reconfigure any custom paths.

### 1.4 Verify the CarMaker Python API installation

Before running the reinforcement learning example, check that Python can successfully import the CarMaker API.

From the repository root, execute:

```powershell
python python/test_connection.py
```

If the API is correctly configured, the script prints:

```text
Hello, CarMaker! API import succeeded.
```

If the test fails, verify that:

- Python 3.6 is active.
- `CARMAKER_PYTHON_PATH` points to the correct CarMaker Python API directory.
- CarMaker 11.0 is properly installed.

**Important:** This script checks that the API can be imported. It does not start a CarMaker simulation or verify the license.

For a more complete connection test, CarMaker also provides example scripts in:

```text
<path-to-CarMaker>/Python/Examples/
```

Configure `runtime_minimal_example.py` to point to an existing CarMaker project and run it:

```powershell
python runtime_minimal_example.py
```

For further information, refer to Chapter 24 of the *CarMaker 11.0 Programmer's Guide*.

### 1.5 Configure the Bernina Pass scenario

The repository includes a CarMaker TestRun for the Bernina Pass:

```text
carmaker/Data/TestRun/berninaPass
```

The simulation uses the following assets from the CarMaker installation:

- Vehicle: `Examples/Demo_Audi_R8`
- Road: `Examples/Measured/Scenario/CHE_BerninaPass.bin`

These assets are not included in the repository and must be available locally.

To verify the simulation:

1. Open CarMaker 11.0.
2. Open the `carmaker/` directory as a CarMaker project.
3. Load the `berninaPass` TestRun.
4. Start the simulation and verify that the scenario runs correctly.

If the project requires additional initialization, use CarMaker's project creation/update functionality.

By default, the Python scripts use the project in the `carmaker/` folder and the `berninaPass` TestRun.

To use another initialized CarMaker project, set:

```powershell
$env:CARMAKER_PROJECT = "C:/path/to/your/project"
$env:CARMAKER_TESTRUN = "berninaPass"
```

### 1.6 Train the example agent

Once the environment is correctly configured and the Bernina Pass scenario runs in CarMaker, the reinforcement learning experiment can be started.

From the repository root, execute:

```powershell
python python/train.py --timesteps 2048 --model models/ppo_carmaker
```

The script uses **Proximal Policy Optimization (PPO)** from Stable-Baselines3 to train an agent that controls the vehicle's throttle.

The example environment, `BerninaEnv`, is defined in `python/bernina_env.py`. It extends the reusable `CarMakerEnv` base class in `python/carmaker_env.py` and uses:

| Component | Description |
|---|---|
| Action | Throttle control between 0.02 and 1.0 |
| Observations | Road distance, simulation time, and vehicle speed |
| Reward | `distance + speed - 0.1 * time` |
| Steering | Controlled by CarMaker's built-in driver |

The trained policy is saved as:

```text
models/ppo_carmaker.zip
```

The number of training timesteps can be modified using the `--timesteps` argument. PPO collects complete rollout batches, so the actual number of steps may exceed the requested minimum.

CarMaker stores simulation logs and results in the project's `SimOutput/` directory.

### 1.7 Evaluate the trained agent

Once training is complete, evaluate the policy using:

```powershell
python python/evaluate.py --model models/ppo_carmaker
```

The script loads the trained model and runs an episode on the Bernina Pass using the actions predicted by the agent.

CarMaker's visualization tools can be used to observe the vehicle during the simulation.

### 1.8 Expected outcome

After completing this setup, you should have a working framework in which a Python reinforcement learning agent can interact with CarMaker, receive vehicle observations, apply throttle actions, and train and evaluate a simple policy.

**The purpose of this example is to validate the CarMaker–RL integration, rather than to train a complete autonomous driving system.**

Once this basic setup is working, it can be used as a starting point for implementing a custom reinforcement learning environment with additional controls, observations, and reward functions.

# Step 2 — Develop your own environment

The next step is to extend the provided example to implement a custom autonomous driving environment.

Create a subclass of `CarMakerEnv` from `python/carmaker_env.py`. The base class handles simulation startup and shutdown, Gym reset/step calls, and simulation timing. It is abstract and cannot be instantiated directly.

Use `python/bernina_env.py` as an example: define `action_space` and `observation_space` in your subclass, implement the asynchronous `set_action` and `get_observation` methods, and implement the synchronous `compute_reward` method. `set_action` should apply the action and wait for the simulation step; the base class then advances `dt` by `time_step`.

Override `is_done(observation)` to add scenario-specific termination conditions. The base implementation checks the time limit; `BerninaEnv` also ends the episode when the car stops after five seconds. Configure the scenario through `python/parametrization_testrun.py` or the `CARMAKER_PROJECT` and `CARMAKER_TESTRUN` environment variables.
