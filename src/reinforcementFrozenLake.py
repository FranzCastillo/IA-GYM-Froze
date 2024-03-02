import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import gymnasium as gym
import numpy as np
import pickle

desc=["SFFF", "FHFH", "FFFH", "HFFG"]
 
def run(episodes, agentMode=True, render=False): # agentMode: true for training, false for evaluation
    # initialize the environment with given parameters
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True, render_mode='human' if render else None)

    # load or initialize q table
    if agentMode:
        
        # initialize q table for training
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        # load q-table from file for evaluation
        with open('frozen_lake4x4.pkl', 'rb') as f:
            q = pickle.load(f)

    # parameters for q learning
    learningRate = 0.9
    discount = 0.9
    epsilon = 0.9
    epsilonDecay = 0.0001
    rng = np.random.default_rng()

    # store rewards for each episode
    episodesReward = np.zeros(episodes)  

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # decide action based on epsilon greedy policy
            if agentMode and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            # update q table
            if agentMode:
                q[state, action] += learningRate * (
                    reward + discount * np.max(q[new_state,:]) - q[state, action]
                )

            state = new_state

        # update epsilon and learning rate
        epsilon = max(epsilon - epsilonDecay, 0)
        if epsilon == 0:
            learningRate = 0.0001

        # record if episode was successful
        if reward == 1:
            episodesReward[i] = 1

    env.close()

    # calculate and plot the total rewards over time
    totalRewards = np.zeros(episodes)
    for t in range(episodes):
        totalRewards[t] = np.sum(episodesReward[max(0, t-100):(t+1)])
    plt.plot(totalRewards)
    plt.savefig('frozenLake4x4.png')

    # save the q table if training
    if agentMode:
        with open("frozenLake4x4.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(10, agentMode=True, render=True)