from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mia_rl.agents.control.linear_sarsa import LinearSarsaControl
from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
from mia_rl.experiments.control import greedy_path, greedy_policy_from_agent
from mia_rl.experiments.fa_training import train_fa_agent
from mia_rl.features.windy_gridworld import (
    STATE_ACTION_FEATURE_DIM,
    state_action_features,
)
from mia_rl.plots.windy_gridworld import (
    plot_episode_lengths,
    plot_policy,
    plot_td_errors,
    plot_value_heatmap,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

algorithm = "Linear SARSA"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear SARSA on Windy Gridworld.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Step-size.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/windy_gridworld_linear_sarsa",
        help="Directory inside mia_rl where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def run_windy_gridworld_linear_sarsa():

    args = parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "windy_gridworld_linear_sarsa"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = WindyGridworldEnv()

    _phi_cache: dict = {}
    def phi(s, a):
        key = (s, a)
        if key not in _phi_cache:
            _phi_cache[key] = state_action_features(s, a, env)
        return _phi_cache[key]
    agent = LinearSarsaControl(
        actions=ACTIONS,
        phi=phi,
        n_features=STATE_ACTION_FEATURE_DIM,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        seed=args.seed,
    )

    print(f"Training LinearSarsaControl for {args.episodes} episodes...")
    episode_lengths, _, td_error_curve = train_fa_agent(env, agent, args.episodes, max_steps=args.max_steps)

    policy = greedy_policy_from_agent(env, agent)
    path = greedy_path(env, policy)

    plot_episode_lengths(output_dir, algorithm, episode_lengths, title="Linear SARSA: episode length over training")
    plot_td_errors(output_dir, algorithm, td_error_curve, title="Linear SARSA: mean |TD error| per episode")
    plot_value_heatmap(
        output_dir, 
        algorithm, 
        env,
        value_fn=lambda s: max(agent.action_value_of(s, a) for a in ACTIONS),
        title="Linear SARSA: V(s) = max_a q_hat(s, a)",
    )
    plot_policy(output_dir, algorithm, env, policy, path=path, title="Linear SARSA: greedy policy")

    print(f"Saved plots to {output_dir}")
    print(f"Final greedy path length: {len(path) - 1}")

    input("Press Enter to close...")


