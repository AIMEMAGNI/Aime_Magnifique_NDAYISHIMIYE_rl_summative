import os

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# Import your custom environment
from environment.custom_env import MissionEnv  # Import your custom environment

# Create log directories
ppo_log_path = "logs/PPO2"
ppo_model_path = "models/PPO2"
video_path = "videos/PPO2"
os.makedirs(ppo_log_path, exist_ok=True)
os.makedirs(ppo_model_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# Create environment and wrap it with Monitor
ppo_env = MissionEnv()

# Ensure that the environment has a termination condition
if not hasattr(ppo_env, "max_steps"):
    ppo_env.max_steps = 500  # Increased max steps per episode for better learning

ppo_env = Monitor(ppo_env, ppo_log_path)  # Logs rewards and episode lengths
ppo_env = DummyVecEnv([lambda: ppo_env])  # SB3 compatibility

# Updated Hyperparameters for PPO
ppo_agent = PPO(
    policy="MlpPolicy",
    env=ppo_env,
    learning_rate=3e-4,  # Adjusted for stability
    n_steps=2048,  # Increased for better trajectory collection
    batch_size=64,  # Larger batch size for efficiency
    n_epochs=10,  # More updates per batch
    gamma=0.99,  # Discount factor for long-term rewards
    gae_lambda=0.95,  # Smoother advantage estimation
    clip_range=0.2,  # More stable policy updates
    ent_coef=0.01,  # Small entropy bonus for exploration
    verbose=1,
    tensorboard_log=ppo_log_path
)

# Train the model
ppo_agent.learn(total_timesteps=200000)  # Increased training steps

# Save the trained model
ppo_agent.save(os.path.join(ppo_model_path, "ppo_model2"))

# Ensure logs are written before reading them
ppo_env.close()

# Check if monitor file exists and is not empty
monitor_log_file = os.path.join(ppo_log_path, "monitor.csv")

if not os.path.exists(monitor_log_file) or os.stat(monitor_log_file).st_size <= 1:
    raise FileNotFoundError(
        "Monitor file is empty. Ensure episodes are completing with done=True.")

# Load logged rewards and compute moving average
reward_data = pd.read_csv(monitor_log_file, skiprows=1)  # Skip comment header

if "r" not in reward_data.columns:
    raise ValueError(
        "Monitor file does not contain reward data. Check environment reward function.")

reward_data['avg_reward'] = reward_data['r'].rolling(
    window=50, min_periods=1).mean()

# Plot and save the reward progress
plt.figure(figsize=(10, 5))
plt.plot(reward_data.index,
         reward_data['avg_reward'], label="Average Reward", color='b')
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Average Reward Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(ppo_log_path, "PPOreward_plot2.png"))
plt.close()

# ---------------- VIDEO RECORDING ---------------- #
print("Recording video for last 2000 steps...")

# Reload environment for video recording
ppo_env = DummyVecEnv([lambda: MissionEnv()])  # Reinitialize environment
total_timesteps = 200000  # Total timesteps during training
video_length = 2000  # Length of the video to capture

# Video recording trigger adjusted to capture only the last 2000 timesteps
video_env = VecVideoRecorder(
    ppo_env,
    video_path,
    record_video_trigger=lambda step: step >= total_timesteps -
    video_length,  # Start recording for last 2000 timesteps
    video_length=video_length,  # Capture last 2000 steps
    name_prefix="ppo2_last_2000_video"  # Video file name prefix
)

# Reload trained model
ppo_agent = PPO.load(os.path.join(ppo_model_path, "ppo_model2"), env=video_env)

# Run the agent for 2000 steps and record
obs = video_env.reset()
for _ in range(video_length):
    action, _states = ppo_agent.predict(obs)
    obs, rewards, dones, info = video_env.step(action)

# Close the video environment
video_env.close()
print(f"Video saved in {video_path}")
