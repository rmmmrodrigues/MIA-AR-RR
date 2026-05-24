from __future__ import annotations

import matplotlib.pyplot as plt

def plot_feature_encoding(phi, current_player, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.5))
    titles = ["my piece (dim 0,3,6,...)", "opponent piece (dim 1,4,7,...)", "empty (dim 2,5,8,...)"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for k, (ax, title, color) in enumerate(zip(axes, titles, colors)):
        vals = phi[k::3]   # every 3rd element starting at k
        ax.bar(range(9), vals, color=color, edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(9))
        ax.set_xticklabels([f"cell {i+1}" for i in range(9)], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([0, 1])

    fig.suptitle(f"Board encoding from player {'O' if current_player == -1 else 'X'}'s perspective", fontsize=12, y=1.02)
    plt.tight_layout()

    plt.savefig(output_dir / "feature_encoding.png", dpi=150, bbox_inches="tight") 

    plt.show(block=False)  


def plot_learning_curves(checkpoints, results, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(checkpoints, results["win_rates_as_x"],  label="Win as X", color="tab:blue",   linewidth=1.8)
    ax.plot(checkpoints, results["win_rates_as_o"],  label="Win as O", color="tab:orange",  linewidth=1.8)
    ax.plot(checkpoints, results["draw_rates_as_x"], label="Draw as X", color="tab:blue",  linewidth=1, linestyle="--")
    ax.axhline(0.58, color="grey", linestyle=":", linewidth=1, label="random baseline")
    ax.set_xlabel("Episode"); ax.set_ylabel("Rate")
    ax.set_title("Win / draw rate vs random agent")
    ax.legend(fontsize=9); ax.set_ylim(0, 1)

    plt.savefig(output_dir / "learning_curves.png", dpi=150, bbox_inches="tight") 

    plt.show(block=False)     
