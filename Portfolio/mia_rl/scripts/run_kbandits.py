#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://www.youtube.com/watch?v=e3L4VocZnnQ
# https://www.youtube.com/watch?v=Zgwfw3bzSmQ

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mia_rl.plots.kbandits import plot_epsilon_greedy, plot_gradient_bandit, plot_optimistic_vs_ucb


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    output_dir = PACKAGE_ROOT / "outputs" / "kbandits"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plot_epsilon_greedy()
    plt.savefig(output_dir / "epsilon_greedy.png", dpi=150, bbox_inches="tight")

    plt.figure()
    plot_optimistic_vs_ucb()
    plt.savefig(output_dir / "optimistic_vs_ucb.png", dpi=150, bbox_inches="tight")

    plt.figure()
    plot_gradient_bandit()
    plt.savefig(output_dir / "gradient_bandit.png", dpi=150, bbox_inches="tight")

    print(f"Saved plots to {output_dir}")
    plt.show()
