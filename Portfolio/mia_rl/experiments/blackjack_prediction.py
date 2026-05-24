from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mia_rl.agents.prediction import FirstVisitMonteCarloPrediction, TD0Prediction, TDNPrediction
from mia_rl.envs.blackjack import BlackjackEnv
from mia_rl.experiments.training import generate_episode, train_prediction_agent
from mia_rl.plots.blackjack import plot_value_difference, plot_value_function
from mia_rl.policies.blackjack import ThresholdPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Blackjack model-free prediction experiments.")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes for each algorithm.")
    parser.add_argument("--td-alpha", type=float, default=0.05, help="Step-size for TD(0).")
    parser.add_argument("--threshold", type=int, default=20, help="Policy threshold: hit below this sum.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="outputs/blackjack_prediction", help="Directory inside mia_rl where plots will be saved.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()

def run_blackjack_prediction():
    args = parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "blackjack_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = ThresholdPolicy(threshold=args.threshold)

    sample_env = BlackjackEnv(seed=args.seed)
    sample_episode = generate_episode(sample_env, policy)
    print(f"Sample episode length: {len(sample_episode.transitions)}")
    print("First transitions:")
    for transition in sample_episode.transitions[:5]:
        print(transition)

    mc_env = BlackjackEnv(seed=args.seed)
    td_env = BlackjackEnv(seed=args.seed)
    tdn_env = BlackjackEnv(seed=args.seed)

    mc_agent = FirstVisitMonteCarloPrediction(gamma=1.0)
    td_agent = TD0Prediction(alpha=args.td_alpha, gamma=1.0)
    tdn_agent = TDNPrediction(n=5, alpha=args.td_alpha, gamma=1.0)

    checkpoints = sorted({cp for cp in (1000, 5000, args.episodes) if cp <= args.episodes})

    print(f"Training First-Visit Monte Carlo for {args.episodes} episodes...")
    mc_history = train_prediction_agent(mc_env, policy, mc_agent, args.episodes, checkpoints=checkpoints)

    print(f"Training TD(0) for {args.episodes} episodes...")
    td_history = train_prediction_agent(td_env, policy, td_agent, args.episodes, checkpoints=checkpoints)

    
    print(f"Training TD({tdn_agent.n}) for {args.episodes} episodes...")
    tdn_history = train_prediction_agent(tdn_env, policy, tdn_agent, args.episodes, checkpoints=checkpoints)
    final_mc = mc_history[args.episodes]
    final_td = td_history[args.episodes]
    final_tdn = tdn_history[args.episodes]
    
    plot_value_function(output_dir, "blackjack_mc.png", final_mc, title=f"First-Visit MC after {args.episodes} episodes", vmin=-1.0, vmax=1.0)
    plot_value_function(output_dir, "blackjack_td0.png", final_td, title=f"TD(0) after {args.episodes} episodes", vmin=-1.0, vmax=1.0)
    plot_value_difference(output_dir, "blackjack_td_minus_mc.png", final_td, final_mc, title="TD(0) - First-Visit MC", vmin=-1.0, vmax=1.0)
    plot_value_function(output_dir, "blackjack_tdn.png", final_tdn, title=f"TD({tdn_agent.n}) after {args.episodes} episodes", vmin=-1.0, vmax=1.0)
    plot_value_difference(output_dir, "blackjack_tdn_minus_mc.png", final_tdn, final_mc, title=f"TD({tdn_agent.n}) - MC", vmin=-1.0, vmax=1.0)
    plot_value_difference(output_dir, "blackjack_tdn_minus_td0.png", final_tdn, final_td, title=f"TD({tdn_agent.n}) - TD(0)", vmin=-1.0, vmax=1.0)
    
    print(f"Saved plots to {output_dir}")
    
    input("Press Enter to close...")