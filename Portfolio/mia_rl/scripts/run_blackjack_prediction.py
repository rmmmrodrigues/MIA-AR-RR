from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    from mia_rl.experiments.blackjack_prediction import run_blackjack_prediction
    
    run_blackjack_prediction()

if __name__ == "__main__":
    main()
