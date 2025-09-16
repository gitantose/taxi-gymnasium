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
    gamma= config.DQN_GAMMA, 
    epsilon= config.DQN_EPSILON_START, 
    epsilon_min= config.DQN_EPSILON_MIN, 
    epsilon_decay= config.DQN_EPSILON_DECAY, 
    use_action_mask: bool = config.USE_ACTION_MASK,
    seed = config.SEED,
    lr= config.DQN_LR,  # learning rate for Adam optimizer
    batch_size= config.DQN_BATCH_SIZE,
    memory_size= config.DQN_MEMORY_SIZE
):  
    if episodes is None:
        episodes = config.DQN_EPISODES
        
    env = gym.make("Taxi-v3",is_rainy = config.IS_RAINING,fickle_passenger=config.FICKLE_PASSENGER)

    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    if seed > 0:
        np.random.seed(seed)
        env.action_space.seed(seed)  

    # One-hot encode discrete state
    def one_hot_encode(s):
        state_vec = np.zeros(state_size)
        state_vec[s] = 1.0
        return state_vec

    # Initialize policy network
    policy_net = QNetwork(state_size, action_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Replay buffer
    reply_buffer = deque(maxlen=memory_size)
    rewards = []

    for ep in trange(episodes, desc="Training DQN"):
        if seed > 0:
            state, info = env.reset(seed=config.SEED + ep)
        else:
            state, info = env.reset()
        state_vector = one_hot_encode(state)
        done = False
        total_reward = 0

        while not done:
            valid_actions = np.where(info["action_mask"] == 1)[0]
            # --- Action Selection (ε-greedy) ---
            if np.random.rand() < epsilon:
                if use_action_mask:
                    action = env.action_space.sample(info["action_mask"]) # explore
                else: 
                    action = env.action_space.sample()
            else:
                if use_action_mask:
                    q_vals = policy_net(torch.FloatTensor(state_vector))
                    with torch.no_grad():
                        action = valid_actions[q_vals[valid_actions].argmax().item()] # exploit best action
                else:
                    with torch.no_grad():
                        action = q_vals.argmax().item() # exploit best action

            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_state_vector = one_hot_encode(next_state)

            # Store transition in replay buffer
            reply_buffer.append((state_vector, action, reward, next_state_vector, done,next_info["action_mask"]))
            
            # Move to next state
            state_vector = next_state_vector
            total_reward += reward
            info = next_info

            # Train on mini-batch
            if len(reply_buffer) >= batch_size:
                batch = random.sample(reply_buffer, batch_size)
                states, actions, rewards_, next_states, dones, next_masks = zip(*batch)

                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions)
                rewards_ = torch.FloatTensor(rewards_)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones)

                # Compute the current Q(s, a) for chosen actions
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                # Target: r + γ * max_a’ Q(s’, a’), unless terminal
                if use_action_mask:
                    next_q_values = []
                    with torch.no_grad():
                        q_next_all = policy_net(next_states)
                        for i, mask in enumerate(next_masks):
                            valid_a = np.where(mask == 1)[0]
                            if len(valid_a) > 0:
                                next_q_values.append(q_next_all[i][valid_a].max().item())
                            else:
                                next_q_values.append(0.0)
                    next_q_values = torch.FloatTensor(next_q_values)
                else:
                    next_q_values = policy_net(next_states).max(1)[0]


                target = rewards_ + gamma * next_q_values * (1 - dones)

                # Loss = MSE(Q_predicted, Q_target)
                loss = criterion(q_values, target.detach())

                # Set all the network parameters to 0
                optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Update the network weights with the optimizer
                optimizer.step()

        # Decay epsilon after each episode
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(total_reward)

    env.close()
    return policy_net, rewards
