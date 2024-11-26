# Breakout RL Agent

This project trains and evaluates a Deep Q-Network (DQN) agent to play Atari's **Breakout** game using **Stable Baselines3** and **Gym**.

## Overview

The project includes two scripts:
- **`train.py`**: Trains a DQN agent on the Atari Breakout game and saves the trained policy network.
- **`play.py`**: Loads the saved policy network and plays the game using the trained agent.

---

## Features

- Implements a **DQN agent** using Stable Baselines3 with a convolutional neural network (CNN) policy.
- Trains the agent for image-based inputs (suitable for Atari games).
- Saves the trained policy as a `.h5` file.
- Visualizes the agent's gameplay in real-time.

---

## Setup

### Prerequisites

1. **Python 3.8 or above**  
   Ensure you have Python installed on your system.

2. **Install Dependencies**  
   Run the following command to install the required libraries:
   ```bash
   pip install gymnasium stable-baselines3 ale-py
   ```

3. **Atari ROMs**  
   The ALE (Arcade Learning Environment) requires Atari ROMs. You can install them by running:
   ```bash
   pip install ale-py[roms]
   ```

### Clone the Repository

1. Clone this repository:
   ```bash
   git clone formative-deep-q-learning
   cd formative-deep-q-learning
   ```

---

## Usage

### Training the Agent

1. Run the `train.py` script:
   ```bash
   python train.py
   ```

2. The training process will save the model as `policy.h5`.  
   Training can take a significant amount of time depending on the number of timesteps and your system's processing power.

### Playing the Game

1. Run the `play.py` script to load the trained model and visualize gameplay:
   ```bash
   python play.py
   ```

2. The script will display the game window and show the agent playing the game.  
   You can modify the number of episodes to play by editing the `episodes` variable in the script.

---

## Code Overview

### `train.py`
- Sets up the **Breakout-v5** environment using Gym.
- Defines a **DQN agent** with a convolutional neural network policy (`CnnPolicy`) and custom hyperparameters.
- Trains the agent for 1,000,000 timesteps (modifiable in `model.learn()`).
- Saves the trained model as `policy.h5`.

### `play.py`
- Loads the saved policy model (`policy.h5`).
- Sets up the **Breakout-v5** environment for real-time visualization.
- Plays the game for a specified number of episodes using a **GreedyQPolicy** (deterministic action selection).
- Displays the game window with real-time rendering.

---

## Key Technologies

- **Gymnasium (ALE/Breakout-v5)**: Simulation environment for Atari Breakout.
- **Stable Baselines3**: High-level library for RL agents.
- **ALE-Py**: Library for Arcade Learning Environment.

---

## Notes

- Ensure your system supports **rendering** for `play.py` to display the game window.
  - For headless servers or virtual machines, use an X server (e.g., Xvfb).
- Training duration depends on the specified timesteps and system resources.
- You can adjust the hyperparameters in `train.py` to experiment with the agent's performance.

---

## Author

Larissa Bizimungu
