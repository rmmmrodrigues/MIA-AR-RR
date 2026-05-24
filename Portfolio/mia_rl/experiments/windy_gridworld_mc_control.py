from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mia_rl.agents.control import MonteCarloControl
from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
from mia_rl.experiments.control import greedy_path, greedy_policy_from_agent, train_control_agent
from mia_rl.plots.windy_gridworld import plot_episode_lengths, plot_episode_rewards, plot_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


algorithm = "MC Control" 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Windy Gridworld with Monte Carlo control.")
    parser.add_argument("--episodes", type=int, default=1_000, help="Number of training episodes.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate for epsilon-greedy control.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--max-steps", type=int, default=1_000, help="Maximum steps per episode before truncation.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/windy_gridworld_mc_control",
        help="Directory inside mia_rl where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def run_windy_gridworld_mc_control():

    args = parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "windy_gridworld_mc_control"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = WindyGridworldEnv()
        agent = MonteCarloControl(actions=ACTIONS, epsilon=args.epsilon, gamma=args.gamma, seed=args.seed)

        episode_lengths, episode_rewards = train_control_agent(env, agent, args.episodes, max_steps=args.max_steps)
        policy = greedy_policy_from_agent(env, agent)
        path = greedy_path(env, policy)

        plot_episode_lengths(output_dir, algorithm, episode_lengths, title="MC control: episode length over training")
        plot_episode_rewards(output_dir, algorithm, episode_rewards, title="MC control: episode reward over training")
        plot_policy(output_dir, algorithm, env, policy, path=path, title="Windy Gridworld greedy policy after MC control training")

        print(f"Saved plots to {output_dir}")
        print(f"Final greedy path length: {len(path) - 1}")

    except NotImplementedError as exc:
        print("\nThis practical is not complete yet.")
        print("Please finish the TODOs in:")
        print("- mia_rl/envs/windy_gridworld.py")
        print(f"\nOriginal message: {exc}")
        return

    input("Press Enter to close...")

