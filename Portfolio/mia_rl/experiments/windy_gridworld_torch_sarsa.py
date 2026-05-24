"""Run PyTorch SARSA with linear function approximation on Windy Gridworld.

Section 3 of the function-approximation class:
  Implements semi-gradient SARSA in two PyTorch modes:

  Manual update (use_optimizer=False):
      with torch.no_grad():
          w += alpha * delta * phi(s, a)
      Equivalent to LinearSarsaControl — no autograd involved.

  Optimizer update (use_optimizer=True):
      loss = 0.5 * (target.detach() - q_hat(s, a)) ** 2
      loss.backward()
      optimizer.step()
      The target MUST be detached to enforce semi-gradient.
      With 0.5 * MSE and SGD(lr=alpha), the update is identical to the
      manual version: w -= alpha * (q_hat - target) * phi = w += alpha * delta * phi.

The script produces a comparison plot of episode lengths across:
  - Tabular SARSA (from previous class)
  - Linear SARSA (NumPy)
  - Torch SARSA (manual)
  - Torch SARSA (optimizer)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from mia_rl.agents.control import SarsaControl
from mia_rl.agents.control.linear_sarsa import LinearSarsaControl
from mia_rl.agents.control.torch_sarsa import TorchSarsaControl
from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
from mia_rl.experiments.control import greedy_path, greedy_policy_from_agent, train_control_agent
from mia_rl.experiments.fa_training import train_fa_agent
from mia_rl.features.windy_gridworld import STATE_ACTION_FEATURE_DIM, state_action_features
from mia_rl.plots.windy_gridworld import plot_length_comparison, plot_td_error_panels, plot_value_heatmaps, plot_policy_comparison

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


algorithm = "Torch Sarsa"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch SARSA on Windy Gridworld.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Step-size for FA agents.")
    parser.add_argument("--tabular-alpha", type=float, default=0.5, help="Step-size for tabular SARSA.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/windy_gridworld_torch_sarsa",
        help="Directory inside mia_rl where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()

def run_windy_gridworld_torch_sarsa():
    args = parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "windy_gridworld_torch_sarsa"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = WindyGridworldEnv()
    # Memoize phi: the grid is tiny (70 states × 4 actions = 280 entries) and the
    # feature function calls env.step_from_state internally, so caching eliminates
    # repeated env calls during action selection and bootstrap computation.
    _phi_cache: dict = {}
    def phi(s, a):
        key = (s, a)
        if key not in _phi_cache:
            _phi_cache[key] = state_action_features(s, a, env)
        return _phi_cache[key]

    common = dict(actions=ACTIONS, alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma, seed=args.seed)
    common_fa = dict(**common, phi=phi, n_features=STATE_ACTION_FEATURE_DIM)

    agents = {
        "Tabular SARSA":           SarsaControl(actions=ACTIONS, alpha=args.tabular_alpha, epsilon=args.epsilon, gamma=args.gamma, seed=args.seed),
        "Linear SARSA (NumPy)":    LinearSarsaControl(**common_fa),
        "Torch SARSA (manual)":    TorchSarsaControl(**common_fa, use_optimizer=False),
        "Torch SARSA (optimizer)": TorchSarsaControl(**common_fa, use_optimizer=True),
    }

    all_lengths: dict[str, list[int]] = {}
    all_td_errors: dict[str, list[float]] = {}
    policies: dict[str, dict] = {}
    paths: dict[str, list] = {}

    for name, agent in agents.items():
        print(f"Training {name} for {args.episodes} episodes...")
        if name == "Tabular SARSA":
            lengths, _ = train_control_agent(env, agent, args.episodes, max_steps=args.max_steps)
            all_lengths[name] = lengths
        else:
            lengths, _, td_errors = train_fa_agent(env, agent, args.episodes, max_steps=args.max_steps)
            all_lengths[name] = lengths
            all_td_errors[name] = td_errors

        policies[name] = greedy_policy_from_agent(env, agent)
        paths[name] = greedy_path(env, policies[name])
        print(f"  final greedy path length: {len(paths[name]) - 1}")

    plot_length_comparison(output_dir, algorithm, all_lengths)
    plot_td_error_panels(output_dir, algorithm, all_td_errors)

    value_grids = {}
    for name, agent in agents.items():
        grid = np.zeros((env.rows, env.cols))
        for row in range(env.rows):
            for col in range(env.cols):
                grid[row, col] = max(agent.action_value_of((row, col), action) for action in ACTIONS)
        value_grids[name] = grid

    all_values = np.concatenate([grid.ravel() for grid in value_grids.values()])
    vmin = float(all_values.min())
    vmax = float(all_values.max())

    plot_value_heatmaps(
        output_dir, 
        algorithm, 
        agents, 
        env, 
        ACTIONS, 
        vmin, vmax
    )

    plot_policy_comparison(
        output_dir, 
        algorithm, 
        policies, 
        env, 
        paths
    )    

    print(f"Saved plots to {output_dir}")

    input("Press Enter to close...")