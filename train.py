import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy
import ale_py
import time

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
        learning_rate=1e-5,
        buffer_size=500000,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        target_update_interval=10000,
        train_freq=4,
        batch_size=64,
        gamma=0.99,
        learning_starts=1000
    )

    # 3. start the timer to track training duration
    start_time = time.time()

    # 4. Train the agent
    print("Training the DQN agent...")
    model.learn(total_timesteps=500000)

    # 5. stop the timer after training
    end_time = time.time()

    # 6. calculate the total training time
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # 7. Save the trained policy
    model.save('policy.h5')
    print("Training completed and model saved as 'policy.h5'.")

if __name__ == "__main__":
    train_agent()