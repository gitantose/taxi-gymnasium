# test.py
import argparse
import numpy as np
import torch
import gymnasium as gym

from tabular_q_learning import train_q_learning
from dqn import QNetwork
from utility import evaluate_tabular, evaluate_dqn, demo_tabular, demo_dqn, plot_learning_curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["tab", "dqn"], required=True,
                        help="Which agent to test: 'tab' (tabular Q-learning) or 'dqn'")
    parser.add_argument("--eval_episodes", type=int, default=100,
                        help="Number of evaluation episodes (default: 100)")
    parser.add_argument("--demo", action="store_true",
                        help="Run a single demo episode after evaluation")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the learning curves from saved files")
    parser.add_argument("--smooth", type=int, default=50,
                        help="Window mobile average (default: 50)")   
    args = parser.parse_args()

    if args.agent == "tab":
        # Load Q-table
        Q = np.load("results/tabular/Q.npy")
        metrics = evaluate_tabular(Q, episodes=args.eval_episodes)
        print("\n=== Tabular Q-learning Evaluation ===")
        print(metrics)
        if args.demo:
            demo_tabular(Q)
        if args.plot:
            plot_learning_curves(save_dir="results", window=args.smooth)

    elif args.agent == "dqn":
        # Load DQN
        env = gym.make("Taxi-v3")
        state_size = env.observation_space.n
        action_size = env.action_space.n
        model = QNetwork(state_size, action_size)
        model.load_state_dict(torch.load("results/dqn/policy_net.pt"))
        model.eval()

        metrics = evaluate_dqn(model, episodes=args.eval_episodes)
        print("\n=== DQN Evaluation ===")
        print(metrics)
        if args.demo:
            demo_dqn(model)
        if args.plot:
            plot_learning_curves(save_dir="results", window=args.smooth)
