from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    
    from mia_rl.experiments.windy_gridworld_mc_control import run_windy_gridworld_mc_control

    run_windy_gridworld_mc_control()

if __name__ == "__main__":
    main()