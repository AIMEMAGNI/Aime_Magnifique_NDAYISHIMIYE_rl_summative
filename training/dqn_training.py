import os

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from environment.custom_env import MissionEnv

# Ensure environment is defined (replace `MissionEnv()` with your actual environment)
# from your_environment_file import MissionEnv  # Uncomment this if MissionEnv is imported from a separate file

# Create log directories
log_dir = "logs/DQN2"
model_dir = "models/DQN2"
video_dir = "videos"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Create environment and wrap it with Monitor
env = MissionEnv()  # Replace this with your custom environment

# Ensure the environment has a termination condition
if not hasattr(env, "max_steps"):
    env.max_steps = 200  # Default max steps per episode (if missing)

env = Monitor(env, log_dir)  # Logs rewards and episode lengths
env = DummyVecEnv([lambda: env])  # SB3 compatibility

# Updated Hyperparameters
dqn_model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=2.5e-4,
    buffer_size=100000,
    learning_starts=5000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log=log_dir
)

# Train the model
dqn_model.learn(total_timesteps=100000)  # Increased training steps

# Save the trained model
model_path = os.path.join(model_dir, "dqn_model2")
dqn_model.save(model_path)

# Ensure logs are written before reading them
env.close()

# Check if monitor file exists and is not empty
monitor_file = os.path.join(log_dir, "monitor.csv")

if not os.path.exists(monitor_file) or os.stat(monitor_file).st_size <= 1:
    raise FileNotFoundError(
        "Monitor file is empty. Ensure episodes are completing with done=True.")

# Load logged rewards and compute moving average
df = pd.read_csv(monitor_file, skiprows=1)  # Skip comment header

if "r" not in df.columns:
    raise ValueError(
        "Monitor file does not contain reward data. Check environment reward function.")

df['avg_reward'] = df['r'].rolling(window=50, min_periods=1).mean()

# Plot and save the reward progress
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['avg_reward'], label="Average Reward", color='b')
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Average Reward Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, "DQNreward_plot2.png"))
plt.close()

# ---------------- VIDEO CAPTURE ---------------- #
# Reload trained model
env = DummyVecEnv([lambda: Monitor(MissionEnv(), log_dir)]
                  )  # Reload environment with Monitor
dqn_model = DQN.load(model_path, env=env)

# Wrap the environment for video recording (Only save the last video for the last 1000 timesteps)
video_env = VecVideoRecorder(
    env,
    video_dir,
    record_video_trigger=lambda step: step >= 100000 -
    2000,  # Start recording for the last 1000 steps
    video_length=1000,  # Capture the last 1000 steps
    name_prefix="dqn2_last_video"
)

# Reset environment and start recording
obs = video_env.reset()
for _ in range(1000):  # Run for the last 1000 timesteps
    action, _ = dqn_model.predict(obs, deterministic=True)
    obs, rewards, done, info = video_env.step(action)
    if done:
        obs = video_env.reset()  # Reset if episode ends

# Close video recorder (it will save only the last video)
video_env.close()

# Close environment
env.close()

print("Training and video capture completed successfully.")
