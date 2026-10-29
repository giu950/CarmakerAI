# Reinforcement Learning for Autonomous Driving with CarMaker

Licensed under the [MIT License](LICENSE).

This repository provides a minimal working setup for developing autonomous driving agents using **Reinforcement Learning (RL)** and **IPG CarMaker**. 

To verify the installation, the framework includes an example based on the **Bernina Pass scenario**, where a PPO agent controls **only throttle**, while CarMaker's built-in driver handles steering.

---

## 1. Setup and Installation

### Requirements
- **OS:** Windows
- **Simulator:** IPG CarMaker 11.0 (with a valid license)
- **Environment:** Conda (Python 3.6 legacy stack)

> **Dependency Note:** `requirements.txt` pins Gym 0.19.0, Stable-Baselines3 1.3.0, PyTorch 1.10.2, and NumPy 1.19.5 for CarMaker 11.0 compatibility.

### Installation Steps

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/giu950/CarmakerAI.git
   cd CarmakerAI
   ```

2. **Create and configure the Conda environment:**
   ```powershell
   conda create -n CMai python=3.6 pip=21.3.1 -y
   conda activate CMai
   python -m pip install setuptools==59.6.0 wheel==0.37.1
   python -m pip install -r requirements.txt
   ```

3. **Configure the CarMaker Python API:**
   Add your CarMaker API path to your environment variables if it differs from the default:
   ```powershell
   $env:CARMAKER_PYTHON_PATH = "C:/IPG/carmaker/win64-11.0/Python/python3.6"
   ```

4. **Verify the connection:**
   ```powershell
   python tests/test_connection.py
   ```
   *Expected output:* `Hello, CarMaker! API import succeeded.`

---

## 2. Running the Example (Bernina Pass)

1. **Verify the scenario in CarMaker:**
   - Open CarMaker 11.0 and open the `carmaker/` folder as a project.
   - Load and test the `berninaPass` TestRun.

2. **Train the PPO agent:**
   ```powershell
   python python/train.py --timesteps 2048 --model models/ppo_carmaker
   ```

3. **Evaluate the trained policy:**
   ```powershell
   python python/evaluate.py --model models/ppo_carmaker
   ```

---

## 3. Developing a Custom Environment

To build your own agent, subclass `CarMakerEnv` from `python/carmaker_env.py`. Use `python/bernina_env.py` as a reference:

- **`action_space` & `observation_space`**: Define your custom spaces.
- **`set_action` & `get_observation`**: Implement asynchronous communication methods.
- **`compute_reward`**: Implement your reward logic synchronously.
- **`is_done` & `scenario_failure`**: Override terminal conditions and failure detection.

Custom projects can be configured via environment variables:
```powershell
$env:CARMAKER_PROJECT = "C:/path/to/your/project"
$env:CARMAKER_TESTRUN = "your_testrun_name"
```

---

## Project Status & Limitations

- **Legacy Stack:** Relies on Python 3.6 and specific legacy dependencies required for IPG CarMaker 11.0 integration.
- **Single-Control Example:** The provided example demonstrates longitudinal control (throttle only); lateral control (steering) is handled by CarMaker's driver.
- **Licensing:** Requires a valid local IPG CarMaker license to execute simulations.
- **Maintenance:** Maintained primarily as a portfolio project and reference implementation for CarMaker–RL integration.