import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import config
import gymnasium as gym
from collections import deque
from tqdm import trange

# --- Neural Network Architecture for Q-function ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size) # one output per action
        )

    def forward(self, x):
        return self.net(x)

def train_dqn (
    episodes= None,
    gamma= config.DQN_GAMMA, # discount factor
    epsilon= config.DQN_EPSILON_START, # initial exploration
    epsilon_min= config.DQN_EPSILON_MIN, # min exploration
    epsilon_decay= config.DQN_EPSILON_DECAY, # epsilon decay rate
    lr= config.DQN_LR,  # learning rate for Adam optimizer
    batch_size= config.DQN_BATCH_SIZE,
    memory_size= config.DQN_MEMORY_SIZE
):  
    if episodes is None:
        episodes = config.DQN_EPISODES
        
    env = gym.make("Taxi-v3")

    # Discrete state space → we encode each state as a one-hot vector
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # One-hot encode discrete state
    def preprocess_state(s):
        state_vec = np.zeros(state_size)
        state_vec[s] = 1.0
        return state_vec

    # Initialize policy network
    policy_net = QNetwork(state_size, action_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Replay buffer for experience tuples (s, a, r, s’, done)
    memory = deque(maxlen=memory_size)
    rewards = []

    for ep in trange(episodes, desc="Training DQN"):
        state, _ = env.reset()
        state = preprocess_state(state)
        done = False
        total_reward = 0

        while not done:
            # --- Action Selection (ε-greedy) ---
            if np.random.rand() < epsilon:
                action = env.action_space.sample() # explore
            else:
                with torch.no_grad():
                    action = policy_net(torch.FloatTensor(state)).argmax().item() # exploit best action

            # --- Environment Step ---
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_state)

            # Store transition in replay buffer
            memory.append((state, action, reward, next_state, done))
            
            # Move to next state
            state = next_state
            total_reward += reward

            # Train on mini-batch
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards_, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards_ = torch.FloatTensor(rewards_)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                # Current Q(s, a) for chosen actions
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                # Target: r + γ * max_a’ Q(s’, a’), unless terminal
                next_q_values = policy_net(next_states).max(1)[0]
                target = rewards_ + gamma * next_q_values * (1 - dones)

                # Loss = MSE(Q_predicted, Q_target)
                loss = criterion(q_values, target.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon after each episode
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(total_reward)

    env.close()
    return policy_net, rewards
