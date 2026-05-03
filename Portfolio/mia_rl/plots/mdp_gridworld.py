from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from mia_rl.envs.gridworld import Gridworld

ARROW = {"U":"↑", "D":"↓", "L":"←", "R":"→", "·":"·"}

def plot_grid_values_and_policy(
    env: Gridworld,
    V: np.ndarray,
    policy: Optional[Dict[Tuple[int,int], str]] = None,
    title: str = "",
    value_fmt: str = "{:.2f}",
):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title(title)

    # draw grid lines
    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(0, env.n_rows)
    ax.set_xticks(np.arange(env.n_cols+1))
    ax.set_yticks(np.arange(env.n_rows+1))
    ax.grid(True)
    ax.invert_yaxis()  # row 0 at top
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # highlight terminal states
    for (r,c) in env.terminal_states:
        rect = plt.Rectangle((c, r), 1, 1, fill=True, alpha=0.15)
        ax.add_patch(rect)

    # put values + arrows
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = (r,c)
            v = V[r,c]
            txt = value_fmt.format(v)
            ax.text(c+0.5, r+0.45, txt, ha="center", va="center", fontsize=12)
            if policy is not None:
                a = policy[s] if s in policy else "·"
                ax.text(c+0.5, r+0.78, ARROW.get(a,"·"), ha="center", va="center", fontsize=18)

    plt.show()
    return fig, ax