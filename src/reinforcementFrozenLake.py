import pickle

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def run(env, episodes, agentMode=True, render=False):
    """
    Run the frozen lake environment with given parameters
    :param episodes:  number of episodes
    :param agentMode: true for training, false for evaluation
    :param render:  true for visualization, false for no visualization
    :return:  None
    """
    # Hyperparameters for q learning
    learning_rate = 0.9
    discount = 0.9
    epsilon = 0.9
    epsilon_decay = 0.0001
    rng = np.random.default_rng()

    # Load or initialize Q table
    if agentMode:
        # Initialize q table for training
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        # Load q-table from file for evaluation
        with open('src\\frozenLake4x4.pkl', 'rb') as f:
            q = pickle.load(f)

    # Store rewards for each episode
    episodes_reward = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # Decide action based on epsilon greedy policy
            if agentMode and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            # Update q table
            if agentMode:
                q[state, action] += learning_rate * (
                        reward + discount * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        # Update epsilon and learning rate
        epsilon = max(epsilon - epsilon_decay, 0)
        if epsilon == 0:
            learning_rate = 0.0001

        # Record if episode was successful
        if reward == 1:
            episodes_reward[i] = 1

    env.close()

    # Save the q table if training
    if agentMode:
        with open("src\\frozenLake4x4.pkl", "wb") as f:
            pickle.dump(q, f)

def main():
    train = True

    # Initialize the environment with given parameters
    env = gym.make(
        'FrozenLake-v1',
        desc=generate_random_map(size=4, seed=35),
        is_slippery=True,
        render_mode='human' if (not train) else None
    )

    run(
        env,
        1_000_000 if train else 5,  # number of episodes
        agentMode=train,  # agentMode: true for training, false for evaluation
        render=(not train)  # render: true for visualization, false for no visualization
    )

if __name__ == '__main__':
    main()
