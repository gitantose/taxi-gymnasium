import numpy as np
import gymnasium as gym
from tqdm import trange
import config

def train_q_learning(
    episodes= None,
    alpha= config.ALPHA,   
    seed = config.SEED,
    use_action_mask: bool = config.USE_ACTION_MASK,     # learning rate (step size for Q-value updates)
    gamma= config.GAMMA,       # discount factor (importance of future rewards)
    epsilon= config.EPSILON_START,      # exploration rate (ε-greedy policy)
    epsilon_decay= config.EPSILON_DECAY, # Multiplicative decay for epsilon
    epsilon_min= config.EPSILON_MIN # Lower bound for epsilon
):  
    if episodes is None:
        episodes = config.EPISODES
    # --- Initialize Environment ---
    env = gym.make("Taxi-v3",is_rainy = config.IS_RAINING,fickle_passenger=config.FICKLE_PASSENGER)

    # The Taxi environment has:
    # - 500 discrete states (position of taxi, passenger location, destination)
    # - 6 discrete actions (move in 4 directions, pick up, drop off)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    if seed > 0:
        np.random.seed(seed)  
        env.action_space.seed(seed)  

    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions))

    rewards = []

    for ep in trange(episodes, desc="Training Q-Learning"):
        if seed > 0:
            state, info = env.reset(seed=config.SEED + ep)
        else:
            state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            valid_actions = np.nonzero(info["action_mask"] == 1)[0]
            # --- Action Selection (ε-greedy) ---
            if np.random.rand() < epsilon:
                if use_action_mask:
                    action = env.action_space.sample(info["action_mask"]) # explore
                else:
                    action = env.action_space.sample()
            else:
                if use_action_mask:
                    if len(valid_actions) > 0:
                        action = valid_actions[np.argmax(Q[state, valid_actions])]
                    else:
                        action = env.action_space.sample() # exploit (best known action)
                else:
                    # Consider all actions
                    action = np.argmax(Q[state])
                
            # --- Take Action, Observe Next State & Reward ---
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if not (done or truncated):
                if use_action_mask:
                    # Only consider valid next actions for bootstrapping
                    next_mask = info["action_mask"]
                    valid_next_actions = np.nonzero(next_mask == 1)[0]
                    if len(valid_next_actions) > 0:
                        next_max = np.max(Q[next_state, valid_next_actions])
                    else:
                        next_max = 0
                else:
                    # Consider all next actions
                    next_max = np.max(Q[next_state])

                # Update Q-value
                Q[state, action] += alpha * (
                    reward + gamma * next_max - Q[state, action]
                )
            # Move to next state
            state = next_state
        # Decay epsilon (less exploration over time)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(total_reward)

    env.close()
    return Q, rewards
