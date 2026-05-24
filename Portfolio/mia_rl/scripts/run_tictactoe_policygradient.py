import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

    
def main():
    from mia_rl.experiments.tictactoe_policygradient import run_tictactoe_policygradient

    run_tictactoe_policygradient()
   
if __name__ == "__main__":
    main()




