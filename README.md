UR5e Robot Reinforcement Learning This project is a reinforcement learning framework built upon the homestri-ur5e-rl repository. It provides a custom training environment for the UR5e robotic arm using MuJoCo, with support for visual input and training using Proximal Policy Optimization (PPO). The aim of this project is to design and train policies that allow a simulated UR5e robot to interact with randomly generated targets in its workspace.

1. Features Custom UR5e Gymnasium environment (BaseRobot-v0), based on homestri-ur5e-rl. Reinforcement Learning pipeline using Stable-Baselines3 (SB3). PPO algorithm support with image-based observation spaces. Training checkpoint saving and resuming (.zip format). TensorBoard logging for monitoring training metrics. Model testing script with rendering support.

2. Installation Clone this repo: git clone https://github.com/Double7sBurger/Mujoco-ur5e-robot-arm-basic-RL-project.git cd ur5e-rl Create and activate a conda environment: conda create -n mujoco_env python=3.10 -y conda activate mujoco_env Install dependencies: pip install -r requirements.txt Example of key dependencies: mujoco gymnasium stable-baselines3 tensorboard

3. Training Run training with PPO: python scripts/train.py Key options inside train.py: total_timesteps: number of training steps tensorboard_log: path for TensorBoard logging reset_num_timesteps=False: continue training from checkpoints Check training progress with: tensorboard --logdir ./ppo_ur5e_tensorboard/




https://github.com/user-attachments/assets/ce831441-b40d-4ca5-85ca-6da1395610a0

