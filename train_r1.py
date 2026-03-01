import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandapower as pp
import pandapower.networks as nw
from stable_baselines3 import PPO

class TripleThreatEnv(gym.Env):
    def __init__(self):
        super(TripleThreatEnv, self).__init__()
        self.net = nw.case14()
        # Install a massive 50 MW Battery at Bus 4 to stabilize the grid
        pp.create_storage(self.net, bus=4, p_mw=0, max_e_mwh=100, sn_mva=100)
        
        # Action: AI chooses to discharge (-50MW) or charge (+50MW) the battery
        self.action_space = spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

    def step(self, action):
        self.net.storage.p_mw.at[0] = action[0]
        
        try:
            pp.runpp(self.net)
            # Core Physics: Calculate Technical Loss
            tech_loss = self.net.res_line.pl_mw.sum()
            
            # Core Stability: Check Voltage at the weakest node (Bus 14)
            voltage_pu = self.net.res_bus.vm_pu.min()
            
            # The AI Reward Formula
            reward = -tech_loss 
            if voltage_pu < 0.95:
                reward -= 500.0 # MASSIVE penalty for letting voltage collapse
                
            done = False
        except:
            reward = -1000.0 # Grid crashed
            done = True
            
        obs = np.append(self.net.res_bus.p_mw.values, self.net.storage.p_mw.values)
        return obs, reward, done, False, {}

    def reset(self, seed=None):
        self.net = nw.case14()
        pp.create_storage(self.net, bus=4, p_mw=0, max_e_mwh=100, sn_mva=100)
        pp.runpp(self.net)
        return np.append(self.net.res_bus.p_mw.values, self.net.storage.p_mw.values), {}

def train_rl_autopilot():
    print("Initializing Triple-Threat RL Environment...")
    env = TripleThreatEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("Training RL Agent to balance Technical and Stability parameters...")
    model.learn(total_timesteps=15000)
    
    model.save("ppo_grid_autopilot")
    print("Success! RL Autopilot saved as ppo_grid_autopilot.zip")

if __name__ == "__main__":
    train_rl_autopilot()