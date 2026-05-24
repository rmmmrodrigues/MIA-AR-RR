from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import time

from mia_rl.plots.tictactoe_mcts import plot_mcts_tree, plot_mcts_vs_trained_reinforce, plot_mcts_win_rate_vs_random
from mia_rl.envs.tictactoe import TicTacToeEnv
from mia_rl.agents.planning.mcts import MCTSAgent
from mia_rl.experiments.mcts_tictactoe import (
    make_mcts_policy,
    evaluate_vs_random,
    evaluate_mcts_vs_reinforce,
)
from mia_rl.experiments.tictactoe import play_game, play_game_vs_human
from mia_rl.policies.tictactoe import random_action
from mia_rl.agents.control.reinforce import ReinforceAgent
from mia_rl.experiments.reinforce_tictactoe import train, make_reinforce_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run_tictactoe_mcts():

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "tictactoe_mcts"
    output_dir.mkdir(parents=True, exist_ok=True)

    env   = TicTacToeEnv()
    agent = MCTSAgent(n_simulations=1_000)

    mcts_policy = make_mcts_policy(agent)

    print("=" * 40)
    print(" MCTS (X)  vs  Random (O)")
    print("=" * 40)
    play_game(env, mcts_policy, random_action, render=True)

    # ── Run MCTS on a mid-game state and visualise the tree ───────────────────────
    #   X | O | ·
    #   O | X | ·      X has the main diagonal (0→4), one move from winning
    #   · | · | ·      X to move — can X win immediately?

    mid_game_state = (1, -1, 0, -1, 1, 0, 0, 0, 0)   # X=cells 0,4 — O=cells 1,3
    player_to_move = 1                                  # X's turn

    env_viz = TicTacToeEnv()
    env_viz.board = mid_game_state
    env_viz.current_player = player_to_move

    print("Board state:")
    env_viz.render(mid_game_state)
    print(f"\nRunning MCTS ({agent.n_simulations} simulations)…")

    action, root = agent.search(mid_game_state, player_to_move)
    print(f"Chosen action: cell {action + 1}  ({'win!' if action == 8 else '?'})")

    plot_mcts_tree(output_dir, root, title=f"MCTS tree — {agent.n_simulations} simulations, X to move")

    sim_counts  = [10, 50, 100, 200, 500, 1000]
    win_x, win_o, times = [], [], []
    N_EVAL = 300

    for n in sim_counts:
        a = MCTSAgent(n_simulations=n)
        t0 = time.time()
        wx, _, _ = evaluate_vs_random(env, a, n_games=N_EVAL, as_player=1)
        wo, _, _ = evaluate_vs_random(env, a, n_games=N_EVAL, as_player=-1)
        elapsed = time.time() - t0
        win_x.append(wx); win_o.append(wo); times.append(elapsed)
        print(f"n={n:4d}  win_X={wx:.0%}  win_O={wo:.0%}  ({elapsed:.1f}s)")

    plot_mcts_win_rate_vs_random(output_dir, sim_counts, win_x, win_o)

    # Same hyperparameters as TicTacToe_PolicyGradient.ipynb
    SEED            = 42
    NUM_EPISODES    = 100_000
    ALPHA           = 0.02
    GAMMA           = 1.0
    ENTROPY_BETA    = 0.01
    RANDOM_OPP_FRAC = 0.5

    reinforce_agent = ReinforceAgent(alpha=ALPHA, gamma=GAMMA, entropy_beta=ENTROPY_BETA, seed=SEED)
    results = train(
        reinforce_agent,
        num_episodes=NUM_EPISODES,
        random_opp_fraction=RANDOM_OPP_FRAC,
        eval_every=10_000,
        eval_episodes=200,
    )

    print(f"REINFORCE — win as X: {results['win_rates_as_x'][-1]:.1%}  |  win as O: {results['win_rates_as_o'][-1]:.1%}")    

    reinforce_policy = make_reinforce_policy(reinforce_agent, greedy=True)
    mcts_strong      = MCTSAgent(n_simulations=1000)
    N = 300

    # MCTS as X (first mover) vs REINFORCE as O
    w_x, d_x, l_x = evaluate_mcts_vs_reinforce(env, mcts_strong, reinforce_policy, n_games=N, mcts_as_player=1)
    # MCTS as O (second mover) vs REINFORCE as X
    w_o, d_o, l_o = evaluate_mcts_vs_reinforce(env, mcts_strong, reinforce_policy, n_games=N, mcts_as_player=-1)

    print("MCTS (1000 sims) vs trained REINFORCE")
    print(f"  MCTS plays X:  win={w_x:.0%}  draw={d_x:.0%}  loss={l_x:.0%}")
    print(f"  MCTS plays O:  win={w_o:.0%}  draw={d_o:.0%}  loss={l_o:.0%}")

    labels   = ['MCTS as X', 'MCTS as O']
    wins     = [w_x,  w_o]
    draws    = [d_x,  d_o]
    losses   = [l_x,  l_o]

    x = np.arange(len(labels))
    width = 0.25

    plot_mcts_vs_trained_reinforce(output_dir, x, width, wins, draws, losses, labels)

    print("=" * 40)
    print(" MCTS (X)  vs  REINFORCE (O)")
    print("=" * 40)
    play_game(env, mcts_policy, reinforce_policy, render=True)

    play_game_vs_human(env, mcts_policy, human_plays=-1)    