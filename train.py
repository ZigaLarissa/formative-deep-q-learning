import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy
import ale_py

def train_agent():

    # unnecessary but helpful for IDEs
    gym.register_envs(ale_py)
    
    # 1. Set up the Atari Breakout environment
    env = gym.make('ALE/Breakout-v5')

    # 2. Define the DQN agent using CnnPolicy
    model = DQN(
        policy=CnnPolicy,
        env=env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        target_update_interval=10000,
        train_freq=4,
        batch_size=32,
        gamma=0.99,
        learning_starts=1000
    )

    # 3. Train the agent
    print("Training the DQN agent...")
    model.learn(total_timesteps=50000)

    # 4. Save the trained policy
    model.save('policy.h5')
    print("Training completed and model saved as 'policy.h5'.")

if __name__ == "__main__":
    train_agent()
