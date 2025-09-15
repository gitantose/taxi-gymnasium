# main.py
import argparse
import os,random
import numpy as np
import matplotlib.pyplot as plt
import torch
import config
from tabular_q_learning import train_q_learning
from dqn import train_dqn
import gymnasium as gym
from utility import moving_avg, evaluate_dqn, evaluate_tabular, demo_dqn, demo_tabular

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_tab", type=int, default=None, help="Override tabular episodes")
    parser.add_argument("--episodes_dqn", type=int, default=None, help="Override DQN episodes")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--show_plot", action="store_true", help="If set, show plot (blocking)")
    parser.add_argument("--demo", choices=["none", "tab", "dqn"], default="none")
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.save_dir, "tabular"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "dqn"), exist_ok=True)

    # -------- Train Tabular Q-Learning --------
    Q, rewards_tab = train_q_learning(episodes=args.episodes_tab)
    # Save Q-table and rewards
    np.save(os.path.join(args.save_dir, "tabular", "Q.npy"), Q)
    np.save(os.path.join(args.save_dir, "tabular", "rewards.npy"), np.array(rewards_tab))

    # -------- Train DQN --------
    model, rewards_dqn = train_dqn(episodes=args.episodes_dqn)
    # Save model and rewards
    torch.save(model.state_dict(), os.path.join(args.save_dir, "dqn", "policy_net.pt"))
    np.save(os.path.join(args.save_dir, "dqn", "rewards.npy"), np.array(rewards_dqn))

    # -------- Plot learning curves (non-blocking by default) --------
    plt.figure(figsize=(9, 6))

    # curve raw (trasparenti)
    plt.plot(rewards_tab, alpha=0.25, label="Tabular Q-Learning (raw)")
    plt.plot(rewards_dqn, alpha=0.25, label="DQN (raw)")

    # medie mobili
    tab_ma, tab_idx = moving_avg(rewards_tab, window=50)
    dqn_ma, dqn_idx = moving_avg(rewards_dqn, window=50)
    plt.plot(tab_idx, tab_ma, linewidth=2.0, label="Tabular Q-Learning (MA×50)")
    plt.plot(dqn_idx, dqn_ma, linewidth=2.0, label="DQN (MA×50)")

    # linea di riferimento a 0
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Episodes")
    plt.ylabel("Episode Return")
    plt.title("Taxi-v3: Tabular vs DQN (raw + smoothed)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.2)

    # mostra anche la parte positiva sopra 0 (con margine)
    y_min = min(min(rewards_tab), min(rewards_dqn))
    y_max = max(max(rewards_tab), max(rewards_dqn))
    y_max = max(y_max, 20)          # assicura >= +20 sopra lo zero
    plt.ylim(y_min - 25, y_max + 10)

    plot_path = os.path.join(args.save_dir, "learning_curves_smoothed.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    if args.show_plot:
        plt.show()   # blocca finché non chiudi la finestra
    plt.close()

    # -------- Evaluate greedily --------
    tab_metrics = evaluate_tabular(Q, episodes=args.eval_episodes)
    dqn_metrics = evaluate_dqn(model, episodes=args.eval_episodes)

    print("\n=== Evaluation (greedy, no exploration) ===")
    print(f"Tabular: {tab_metrics}")
    print(f"DQN    : {dqn_metrics}")
    print(f"\nSaved plot to: {plot_path}")
    print(f"Saved tabular Q to: {os.path.join(args.save_dir, 'tabular', 'Q.npy')}")
    print(f"Saved DQN weights to: {os.path.join(args.save_dir, 'dqn', 'policy_net.pt')}")

    # -------- Optional demo (will open a window and block while playing) --------
    if args.demo == "tab":
        demo_tabular(Q)
    elif args.demo == "dqn":
        demo_dqn(model)
