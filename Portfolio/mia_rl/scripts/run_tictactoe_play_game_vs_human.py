import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():

    from mia_rl.envs.tictactoe import TicTacToeEnv
    from mia_rl.features.tictactoe import random_action
    from mia_rl.experiments.tictactoe import play_game_vs_human

    env = TicTacToeEnv()
    play_game_vs_human(env, random_action, human_plays=-1)

if __name__ == "__main__":
    main()