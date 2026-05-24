import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    from mia_rl.experiments.mdp_gridworld import run_mdp_gridworld

    run_mdp_gridworld()

if __name__ == "__main__":
    main()