import gymnasium as gym
import homestri_ur5e_rl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

# Create and wrap the environment
def make_env():
    env = gym.make("BaseRobot-v0", render_mode="human")
    return env

env = DummyVecEnv([make_env])

# Create evaluation environment with rendering
eval_env = DummyVecEnv([make_env])

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1, 
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log="./tensorboard_logs/")

# Train the model
print("Starting training...")
total_timesteps = 1000000  # Adjust based on your needs
model.learn(total_timesteps=total_timesteps)

# Test the trained model with rendering
print("Testing trained model...")
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()

env.close()
