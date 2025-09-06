# main.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tabular_q_learning import train_q_learning
from dqn import train_dqn

# ---------- Evaluation helpers (self-contained) ----------
import gymnasium as gym
import torch

def evaluate_tabular(Q, episodes=100):
    env = gym.make("Taxi-v3")
    returns, successes, steps_list = [], 0, []

    for _ in range(episodes):
        s, _ = env.reset()
        done, total, steps = False, 0, 0
        info = {}
        while not done:
            a = int(np.argmax(Q[s]))
            s, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            total += r
            steps += 1
        returns.append(total)
        success = (info.get("is_success") if "is_success" in info else (r == 20))
        successes += int(bool(success))
        steps_list.append(steps)

    env.close()
    return {
        "avg_return": float(np.mean(returns)),
        "success_rate": successes / episodes,
        "avg_steps": float(np.mean(steps_list)),
    }

def evaluate_dqn(policy_net, episodes=100):
    env = gym.make("Taxi-v3")
    state_size = env.observation_space.n

    def one_hot(s):
        v = np.zeros(state_size, dtype=np.float32)
        v[s] = 1.0
        return v

    returns, successes, steps_list = [], 0, []

    policy_net.eval()
    with torch.no_grad():
        for _ in range(episodes):
            s, _ = env.reset()
            sv = one_hot(s)
            done, total, steps = False, 0, 0
            info = {}
            while not done:
                a = policy_net(torch.from_numpy(sv)).argmax().item()
                s, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                sv = one_hot(s)
                total += r
                steps += 1
            returns.append(total)
            success = (info.get("is_success") if "is_success" in info else (r == 20))
            successes += int(bool(success))
            steps_list.append(steps)

    env.close()
    return {
        "avg_return": float(np.mean(returns)),
        "success_rate": successes / episodes,
        "avg_steps": float(np.mean(steps_list)),
    }

def demo_tabular(Q, fps=2):
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    s, _ = env.reset()
    done = False

    fig, ax = plt.subplots()
    img = ax.imshow(env.render())
    ax.axis("off")

    delay = 1.0 / fps
    while not done:
        a = int(np.argmax(Q[s]))
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        frame = env.render()
        img.set_data(frame)
        plt.pause(delay)

    env.close()
    plt.close(fig)

def demo_dqn(model, fps=2):
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    state_size = env.observation_space.n

    def one_hot(s):
        v = np.zeros(state_size, dtype=np.float32)
        v[s] = 1.0
        return v

    s, _ = env.reset()
    done = False

    fig, ax = plt.subplots()
    img = ax.imshow(env.render())
    ax.axis("off")

    delay = 1.0 / fps
    model.eval()
    with torch.no_grad():
        while not done:
            sv = one_hot(s)
            a = model(torch.from_numpy(sv)).argmax().item()
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            frame = env.render()
            img.set_data(frame)
            plt.pause(delay)

    env.close()
    plt.close(fig)

def moving_avg(x, window=50):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x, np.arange(len(x))
    ma = np.convolve(x, np.ones(window, dtype=np.float32)/window, mode="valid")
    idx = np.arange(window - 1, len(x))
    return ma, idx

# ---------- Main ----------
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
    Q, rewards_tab = train_q_learning()
    # Save Q-table and rewards
    np.save(os.path.join(args.save_dir, "tabular", "Q.npy"), Q)
    np.save(os.path.join(args.save_dir, "tabular", "rewards.npy"), np.array(rewards_tab))

    # -------- Train DQN --------
    model, rewards_dqn = train_dqn()
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
