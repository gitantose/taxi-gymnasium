import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import config

def get_action_policy_tabular(use_action_mask,info,n_actions,Q,s):
    if use_action_mask:
        valid_actions = np.nonzero(info["action_mask"] == 1)[0]
        next_action = valid_actions[np.argmax(Q[s, valid_actions])]
    else:
        next_action = np.argmax(Q[s])
    return next_action

def get_action_policy_dqn(use_action_mask,info,policy_net,sv):
    if use_action_mask:
        valid_actions = np.where(info["action_mask"] == 1)[0]
        q_vals = policy_net(torch.FloatTensor(sv))
        with torch.no_grad():
            next_action = valid_actions[q_vals[valid_actions].argmax().item()]
    else:
        with torch.no_grad():
            next_action = q_vals.argmax().item()
    return next_action

def evaluate_tabular(Q, episodes=100):
    env = gym.make("Taxi-v3",is_rainy = config.IS_RAINING,fickle_passenger=config.FICKLE_PASSENGER)
    returns, successes, steps_list = [], 0, []
    use_action_mask = config.USE_ACTION_MASK
    seed = config.SEED

    for _ in range(episodes):
        if seed > 0:
            s, info = env.reset(seed=seed + _)
        else:
            s, info = env.reset()
        done, total, steps = False, 0, 0
        n_actions = env.action_space.n
        while not done:
            a = get_action_policy_tabular(use_action_mask,info,n_actions,Q,s)
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
    env = gym.make("Taxi-v3",is_rainy = config.IS_RAINING,fickle_passenger=config.FICKLE_PASSENGER)
    state_size = env.observation_space.n
    use_action_mask = config.USE_ACTION_MASK
    seed = config.SEED

    def one_hot(s):
        v = np.zeros(state_size, dtype=np.float32)
        v[s] = 1.0
        return v

    returns, successes, steps_list = [], 0, []

    policy_net.eval()
    with torch.no_grad():
        for _ in range(episodes):
            if seed > 0:
                s, info = env.reset(seed=seed + _)
            else:
                s, info = env.reset()
            sv = one_hot(s)
            done, total, steps = False, 0, 0
            while not done:
                a = get_action_policy_dqn(use_action_mask,info,policy_net,sv)
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
    env = gym.make("Taxi-v3", render_mode="rgb_array",is_rainy = config.IS_RAINING,fickle_passenger=config.FICKLE_PASSENGER)
    if config.SEED > 0:
        s, info = env.reset(seed=config.SEED)
    else:
        s, info = env.reset()
    done = False
    n_actions = env.action_space.n
    use_action_mask = config.USE_ACTION_MASK

    fig, ax = plt.subplots()
    img = ax.imshow(env.render())
    ax.axis("off")

    delay = 1.0 / fps
    while not done:
        a = get_action_policy_tabular(use_action_mask,info,n_actions,Q,s)
        s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        frame = env.render()
        img.set_data(frame)
        plt.pause(delay)

    env.close()
    plt.close(fig)

def demo_dqn(policy_net, fps=2):
    env = gym.make("Taxi-v3", render_mode="rgb_array",is_rainy = config.IS_RAINING,fickle_passenger=config.FICKLE_PASSENGER)
    state_size = env.observation_space.n
    use_action_mask = config.USE_ACTION_MASK

    def one_hot(s):
        v = np.zeros(state_size, dtype=np.float32)
        v[s] = 1.0
        return v

    if config.SEED > 0:
        s, info = env.reset(seed=config.SEED)
    else:
        s, info = env.reset()
    done = False

    fig, ax = plt.subplots()
    img = ax.imshow(env.render())
    ax.axis("off")

    delay = 1.0 / fps
    policy_net.eval()
    with torch.no_grad():
        while not done:
            sv = one_hot(s)
            a = get_action_policy_dqn(use_action_mask,info,policy_net,sv)
            s, r, terminated, truncated, info = env.step(a)
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

def plot_learning_curves(save_dir="results", window=50, out_name="learning_curves_smoothed.png"):
    tab_path = os.path.join(save_dir, "tabular", "rewards.npy")
    dqn_path = os.path.join(save_dir, "dqn", "rewards.npy")
    if not (os.path.exists(tab_path) and os.path.exists(dqn_path)):
        print("⚠️  Rewards files non trovati. Lancia prima `python main.py` per generarli.")
        return

    r_tab = np.load(tab_path)
    r_dqn = np.load(dqn_path)

    plt.figure(figsize=(9, 6))
    # raw
    plt.plot(r_tab, alpha=0.25, label="Tabular (raw)")
    plt.plot(r_dqn, alpha=0.25, label="DQN (raw)")
    # smoothed
    tab_ma, tab_idx = moving_avg(r_tab, window=window)
    dqn_ma, dqn_idx = moving_avg(r_dqn, window=window)
    plt.plot(tab_idx, tab_ma, linewidth=2.0, label=f"Tabular (MA×{window})")
    plt.plot(dqn_idx, dqn_ma, linewidth=2.0, label=f"DQN (MA×{window})")

    # linea di riferimento a 0 e limiti che includono parte positiva
    plt.axhline(0, linestyle="--", linewidth=1)
    y_min = float(min(r_tab.min(), r_dqn.min()))
    y_max = float(max(r_tab.max(), r_dqn.max(), 20))
    plt.ylim(y_min - 25, y_max + 10)

    plt.xlabel("Episodi")
    plt.ylabel("Ritorno per episodio")
    plt.title("Taxi-v3: Tabular vs DQN (raw + media mobile)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.2)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Grafico salvato in: {out_path}")