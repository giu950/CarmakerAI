import gym
from gym import spaces
import numpy as np
from parametrization_testrun import cmapi, make_variations
from cmapi import DVADuration
import asyncio
import nest_asyncio  # Import nest_asyncio

class CarMakerEnvV1(gym.Env):
    def __init__(self):
        super(CarMakerEnvV1, self).__init__()

        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()

        # Control throttle only; the CarMaker driver handles steering.
        self.action_space = spaces.Box(low=0.02, high=1.0, shape=(1,), dtype=np.float32)

        # Define observation space: for example, distance, time, velocity
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([np.inf, np.inf, np.inf]), dtype=np.float32)

        # Keep the original simulation timing and episode behavior.
        self.dt = 0.0
        self.time_step = 0.02
        self.maxTime = 890
        self.loop = asyncio.get_event_loop()  # Create an event loop
        print("Init success")

    async def start_simulation(self):
        variations = await make_variations()

        cmapi.logger.info(f"Executing with self.simcontrol...")
        self.variation = variations[0]

        self.simcontrol = cmapi.SimControlInteractive()
        print("OK self.simcontrolInteractive")
        self.simcontrol.set_variation(self.variation)
        print("OK set_variation")
        master = cmapi.CarMaker()
        await self.simcontrol.set_master(master)
        print("OK set_master")
        await self.simcontrol.start_and_connect()
        print("OK start_and_connect")
        await self.simcontrol.start_sim()
        print("OK start_sim")

    async def stop_simulation(self):
        print("Stopping simulation")
        await self.simcontrol.create_simstate_condition(cmapi.ConditionSimState.finished).wait()
        await self.simcontrol.stop_and_disconnect()
        print("Simulation stopped")

    async def set_action(self, action):
        self.simcontrol.simio.dva_write_absolute_value("DM.Gas", float(np.asarray(action).reshape(-1)[0]), DVADuration.permanent)
        condition = self.simcontrol.create_quantity_condition(lambda fTime: fTime > self.dt, "Time")
        await condition.wait()
        
        self.dt = self.dt + self.time_step

    async def get_observation(self):
        distance, = await self.simcontrol.simio.dva_read_async("Car.Road.sRoad")
        time, = await self.simcontrol.simio.dva_read_async("Time")
        velocity, = await self.simcontrol.simio.dva_read_async("Car.v")
        sGas, = await self.simcontrol.simio.dva_read_async("DM.Gas")
        self.simcontrol.simio.dva_release("DM.Gas")
        return np.array([distance, time, velocity], dtype=np.float32)

    def compute_reward(self, observation):
        distance, time, velocity = observation
        reward = distance + velocity - 0.1 * time  # Example: reward for maintaining a target distance
        return reward

    def reset(self):
        # Synchronously start a new simulation
        self.dt = 0.0
        self.loop.run_until_complete(self.start_simulation())  # Run the async function
        print("Simulation started")

        # Get the initial observation
        observation = self.loop.run_until_complete(self.get_observation())  # Run the async function
        print("Observation: ", observation)
        return observation

    def step(self, action):
        # Apply action to CarMaker and get new observation synchronously
        self.loop.run_until_complete(self.set_action(action))  # Run the async function
        observation = self.loop.run_until_complete(self.get_observation())  # Run the async function
        if self.dt % 20 == 0:
            print("Observation: ", observation)

        # Compute reward
        reward = self.compute_reward(observation)

        # Determine if the simulation is done (e.g., time limit reached)
        done = False
        if self.dt > self.maxTime or (round(observation[2], 1) == 0.0 and self.dt > 5.0):
            print("Forcing stop")
            print("Distance: ", round(observation[2], 1))
            done = True

        # Additional info (optional)
        info = {}

        # Automatically stop the simulation if done
        if done:
            print("Reward: ", reward)
            self.close()

        return observation, reward, done, info

    def render(self, mode='human'):
        # Optional: Implement if you want to visualize the simulation
        pass

    def close(self):
        # Synchronously stop the simulation
        self.loop.run_until_complete(self.stop_simulation())  # Run the async function

