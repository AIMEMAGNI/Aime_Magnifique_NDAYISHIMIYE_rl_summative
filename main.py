import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import MissionEnv  # Import your custom environment


def main():
    # Path to the pre-trained PPO model
    model_path = "models/ppo_model2.zip"  # Make sure this is correct

    # Create environment and wrap it with DummyVecEnv
    env = MissionEnv()
    env = DummyVecEnv([lambda: env])

    # Load the trained PPO model
    model = PPO.load(model_path, env=env)

    # Test the environment with the loaded PPO model
    evaluate_model(model, env)


def evaluate_model(model, env):
    """Function to evaluate the PPO model."""
    obs = env.reset()
    total_rewards = 0
    done = False
    step_count = 0

    while not done:
        step_count += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        total_rewards += rewards

        # Optionally, render the environment
        env.render()

    print(f"Test completed in {step_count} steps")
    print(f"Total reward: {total_rewards}")


if __name__ == "__main__":
    main()
