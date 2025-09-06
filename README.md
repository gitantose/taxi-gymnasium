
# ML Project Taxi-v3 (Tabular Q-learning & DQN) done by Pizzi Lorenzo and Antonio Serra

This guide gathers all the ways you can run the project: setup, training, evaluation, demo, and plotting of results.

> Note: the commands assume you are in the project root (the folder containing main.py, test.py, etc.).

---

## 1) Requirements

- Python 3.10+
- Demo uses **matplotlib** con `rgb_array` (no `pygame`).

---

## 2) Setup environment (recommended to use a virtual environment)


# create a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# download requirements
pip install -r requirements.txt


---

## 3) Training **from zero** (tabular + DQN) — `main.py`

It train both agents, saves models and rewards and produces graphs.


# full train + save graphs
python main.py


## Useful options


# show the graph at the end of the training
python main.py --show_plot

# change the number of episodes
python main.py --episodes_tab 7000 --episodes_dqn 1500

# change the output dir (default: results)
python main.py --save_dir my_results

# execute a demo at the end of the training
python main.py --demo tab        # or: --demo dqn

Output:
- `results/tabular/Q.npy` — Q-table 
- `results/tabular/rewards.npy` — rewards for episode (tabular)
- `results/dqn/policy_net.pt` — weights of DQN model
- `results/dqn/rewards.npy` — rewards for episode (DQN)
- `results/learning_curves_smoothed.png` — graph (raw + media mobile)

---

## 4) Evaluation **without training** — `test.py`

Uses files stored in `results/` to do evaluation or demo, **without re-train**.

### 4.1 Valutazione greedy


# evluate the Q-table (greedy, no exploration)
python test.py --agent tab

# evluate the DQN (greedy, no exploration)
python test.py --agent dqn

# change the number of episodes in evaluation (default: 100)
python test.py --agent tab --eval_episodes 200


### 4.2 Visual Demo (matplotlib)


# show a greedy episode using matplotlib (rgb_array)
python test.py --agent tab --demo
python test.py --agent dqn --demo


> You can adjust the demo execution speed in `main.py` in the demo_dqn and demo_tabular functions by changing the fps parameter (es. `fps=4`).

### 4.3 Plot from stored files (without training)

# plot the graph of stored rewards (raw + mobile average)
python test.py --agent tab --plot

# set te window of mobile average (default: 50)
python test.py --agent dqn --plot --smooth 100

The graph is stored in `results/learning_curves_smoothed.png`.

---

## 5) Rapid execution (cheat sheet)

# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train + save + plot (smussato)
python main.py
python main.py --show_plot

# Evaluate (no training)
python test.py --agent tab
python test.py --agent dqn
python test.py --agent tab --eval_episodes 200

# Demo (matplotlib)
python test.py --agent tab --demo
python test.py --agent dqn --demo

# Plot of saved files
python test.py --agent tab --plot
python test.py --agent dqn --plot --smooth 100

---

## 6) Useful Notes

- `main.py` **train always from 0**: to use when parameters of tabular q-lerning or dqn are changed
- `test.py` **not train**: load models/scores stored for evaluation,demo and plots.
- `cofig.py`: contains constants used for q-learning and dqn
- `utility.py`: contains methods to check and test the results getting after the learning
- `tabular_q_learning.py`: contains the logic for tabular q learning
- `dqn.py`: contains the logic for dqn

