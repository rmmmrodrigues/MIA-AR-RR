from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mia_rl.plots.kbandits import plot_epsilon_greedy, plot_gradient_bandit, plot_optimistic_vs_ucb

# ============================================================
# Experiment runner
# ============================================================

def run_kbandits():
    output_dir = PACKAGE_ROOT / "outputs" / "kbandits"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_epsilon_greedy(output_dir)

    plot_optimistic_vs_ucb(output_dir)

    plot_gradient_bandit(output_dir)

    print(f"Saved plots to {output_dir}")

    input("Press Enter to close...")
