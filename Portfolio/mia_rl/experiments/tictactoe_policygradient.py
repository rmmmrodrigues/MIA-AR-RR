import sys
from pathlib import Path

from mia_rl.plots.tictactoe_policygradient import plot_feature_encoding, plot_learning_curves
from mia_rl.envs.tictactoe import TicTacToeEnv
from mia_rl.features.tictactoe import encode_state
from mia_rl.agents.control.reinforce import ReinforceAgent
from mia_rl.experiments.reinforce_tictactoe import (
    train,
    make_reinforce_policy,
    _play_silent,
)
from mia_rl.experiments.tictactoe import play_game, play_game_vs_human

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

    
def run_tictactoe_policygradient():

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "tictactoe_policygradient"
    output_dir.mkdir(parents=True, exist_ok=True)

    #1. Recap — the 27-dimensional feature vector
    #Each board state is encoded **from the current player's perspective**:
    #| Cell content | Encoding |
    #|---|---|
    #| My piece     | `[1, 0, 0]` |
    #| Opponent     | `[0, 1, 0]` |
    #| Empty        | `[0, 0, 1]` |
    #9 cells × 3 dims = **27 features** (`STATE_FEATURE_DIM = 27`).
    #Because the encoding is perspective-relative, **the same policy weights work for both X and O**,
    #which is exactly what we exploit in self-play.
    #Let's visualise the encoding for a concrete board.
    # Example board after three moves
    #   X | 2 | 3
    #   4 | O | 6
    #   7 | 8 | X
    sample_board = (1, 0, 0, 0, -1, 0, 0, 0, 1)
    current_player = -1  # O's turn

    phi = encode_state(sample_board, current_player)

    plot_feature_encoding(phi, current_player, output_dir)

    env = TicTacToeEnv()
    env.board = sample_board
    print("Board:")
    env.render(sample_board)
    print(f"\nFeature vector (27 dims): {phi}")

    #Training a policy with REINFORCE
    SEED               = 42
    NUM_EPISODES       = 100_000
    ALPHA              = 0.02
    GAMMA              = 1.0
    ENTROPY_BETA       = 0.01
    RANDOM_OPP_FRAC    = 0.5   # fraction of episodes vs random opponent (rest = self-play)
    EVAL_EVERY         = 2_000
    EVAL_GAMES         = 500

    env   = TicTacToeEnv()
    agent = ReinforceAgent(alpha=ALPHA, gamma=GAMMA, entropy_beta=ENTROPY_BETA, seed=SEED)
    results = train(
        agent,
        num_episodes=NUM_EPISODES,
        eval_every=EVAL_EVERY,
        eval_episodes=EVAL_GAMES,
        random_opp_fraction=RANDOM_OPP_FRAC,
    )

    print(f"Win rate as X: {results['win_rates_as_x'][-1]:.1%}")
    print(f"Win rate as O: {results['win_rates_as_o'][-1]:.1%}")

    #Learning curves
    checkpoints = results["eval_checkpoints"]

    plot_learning_curves(checkpoints, results, output_dir)

    #5. Watching two trained agents play

    #Both X and O use the **greedy** policy extracted from the trained agent.    
    reinforce_policy = make_reinforce_policy(agent, greedy=True)

    print("=" * 40)
    print(" REINFORCE (X)  vs  REINFORCE (O)")
    print("=" * 40)
    result = play_game(env, reinforce_policy, reinforce_policy, render=True)    

    n = 1_000
    outcomes = {1: 0, -1: 0, 0: 0}
    for _ in range(n):
        outcomes[_play_silent(env, reinforce_policy, reinforce_policy)] += 1

    print(f"Over {n} self-play games:")
    print(f"  X wins: {outcomes[1]/n:.1%}  |  O wins: {outcomes[-1]/n:.1%}  |  Draws: {outcomes[0]/n:.1%}")    

    play_game_vs_human(env, reinforce_policy, human_plays=-1)

    input("Press Enter to close...")