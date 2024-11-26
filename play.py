import gymnasium as gym
from stable_baselines3 import DQN
import ale_py
import time

def play_agent():

    # unnecessary but helpful for IDEs
    gym.register_envs(ale_py)

    # 1. Load the trained model
    model = DQN.load('policy.h5')
    print("Model loaded successfully.")

    # 2. Set up the Breakout environment
    env = gym.make('ALE/Breakout-v5', render_mode="human")
    obs = env.reset()

    # 3. Play the game using GreedyQPolicy
    print("Playing with the trained agent...")
    episodes = 5  # Number of episodes to play

    for episode in range(episodes):
        obs, info = env.reset()  # Reset the environment
        done = False
        total_reward = 0

        while not done:
            # Use the trained model to predict the best action
            action, _states = model.predict(obs, deterministic=True)  # Greedy policy
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

            # Render the environment in real-time
            time.sleep(0.03)  # Slight delay for better visualization
            env.render()

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    env.close()  # Close the environment
    print("Game playback completed.")

if __name__ == "__main__":
    play_agent()