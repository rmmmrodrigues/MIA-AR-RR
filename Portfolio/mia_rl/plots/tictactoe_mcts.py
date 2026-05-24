from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_mcts_tree(output_dir, root, title="MCTS search tree — root children"):
    """Visualise the first level of the MCTS tree (root + its children).

    Each subplot shows one possible move with the board state, visit count N,
    Q-value (win rate from the root player's perspective), and UCB1 score.
    The chosen action (most-visited child) is highlighted with a green border.
    """
    import math

    children = sorted(root.children.values(), key=lambda n: n.visit_count, reverse=True)
    n_children = len(children)
    best_action = max(root.children.values(), key=lambda n: n.visit_count).action
    c = math.sqrt(2)  # UCB exploration constant

    cols = min(n_children, 5)
    rows = math.ceil(n_children / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 3.0))
    axes = np.array(axes).flatten()

    SYMBOLS = {0: "·", 1: "X", -1: "O"}
    COLORS  = {0: "white", 1: "#4a90d9", -1: "#e07b39"}

    for ax, child in zip(axes, children):
        board = child.state

        # Draw 3×3 grid
        for r in range(3):
            for col_idx in range(3):
                idx = r * 3 + col_idx
                val = board[idx]
                rect = plt.Rectangle(
                    [col_idx, 2 - r], 1, 1,
                    facecolor=COLORS[val], edgecolor="black", linewidth=0.8
                )
                ax.add_patch(rect)
                ax.text(
                    col_idx + 0.5, 2 - r + 0.5, SYMBOLS[val],
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if val != 0 else "#aaaaaa"
                )

        # Highlight the played cell
        pr, pc = divmod(child.action, 3)
        ax.add_patch(plt.Rectangle(
            [pc, 2 - pr], 1, 1,
            facecolor="none", edgecolor="#ffdd00", linewidth=2.5
        ))

        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect("equal")
        ax.axis("off")

        # Stats
        N = child.visit_count
        Q = (-child.value_sum / N) if N > 0 else 0.0   # from root's perspective
        ucb = child.ucb(c) if N > 0 else float("inf")
        ucb_str = f"{ucb:.5f}" if ucb != float("inf") else "∞"

        chosen = child.action == best_action
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#2ecc71" if chosen else "#cccccc")
            spine.set_linewidth(3.5 if chosen else 1.0)

        label = f"cell {child.action + 1}  {'← chosen' if chosen else ''}\nN={N}  Q={Q:+.4f}\nUCB={ucb_str}"
        ax.set_title(label, fontsize=8, color="#2ecc71" if chosen else "black", pad=4)

    for ax in axes[n_children:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "mcts_tree.png")
    plt.show(block=False)


def plot_mcts_win_rate_vs_random(output_dir, sim_counts, win_x, win_o):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sim_counts, win_x, marker='o', label='Win as X', color='tab:blue',   linewidth=1.8)
    ax.plot(sim_counts, win_o, marker='s', label='Win as O', color='tab:orange',  linewidth=1.8)
    ax.axhline(0.58, color='grey', linestyle=':', linewidth=1, label='random baseline')
    ax.set_xlabel('Number of simulations'); ax.set_ylabel('Win rate vs random')
    ax.set_title('MCTS win rate vs random opponent')
    ax.legend(); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "mcts_win_rate_vs_random.png")
    plt.show(block=False)

def plot_mcts_vs_trained_reinforce(output_dir, x, width, wins, draws, losses, labels):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width, wins,   width, label='Win',  color='tab:green')
    ax.bar(x,         draws,  width, label='Draw', color='tab:grey')
    ax.bar(x + width, losses, width, label='Loss', color='tab:red')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Rate')
    ax.set_ylim(0, 1.05)
    ax.set_title('MCTS (1000 sims) vs trained REINFORCE')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mcts_vs_trained_reinforce.png")
    plt.show(block=False)    