import numpy as np
import gymnasium as gym
from tqdm import trange
import config

def train_q_learning(
    episodes= None,
    alpha= config.ALPHA,        # learning rate (step size for Q-value updates)
    gamma= config.GAMMA,       # discount factor (importance of future rewards)
    epsilon= config.EPSILON_START,      # exploration rate (ε-greedy policy)
    epsilon_decay= config.EPSILON_DECAY, # Multiplicative decay for epsilon
    epsilon_min= config.EPSILON_MIN # Lower bound for epsilon
):  
    if episodes is None:
        episodes = config.EPISODES
    # --- Initialize Environment ---
    env = gym.make("Taxi-v3")

    # The Taxi environment has:
    # - 500 discrete states (position of taxi, passenger location, destination)
    # - 6 discrete actions (move in 4 directions, pick up, drop off)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions))

    rewards = []

    for ep in trange(episodes, desc="Training Q-Learning"):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # --- Action Selection (ε-greedy) ---
            if np.random.rand() < epsilon:
                action = env.action_space.sample() # explore
            else:
                action = np.argmax(Q[state]) # exploit (best known action)

            # --- Take Action, Observe Next State & Reward ---
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # --- Q-learning Update Rule ---
            # Q(s,a) ← Q(s,a) + α [r + γ * max_a' Q(s', a') - Q(s,a)]
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            # Move to next state
            state = next_state
            total_reward += reward

        # Decay epsilon (less exploration over time)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(total_reward)

    env.close()
    return Q, rewards
