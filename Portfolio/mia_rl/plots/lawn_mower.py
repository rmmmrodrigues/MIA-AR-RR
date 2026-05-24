from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from mia_rl.envs.lawn_mower import LawnMowerEnv

# =========================================================
# PLOT HEATMAP + TRAJECTORY
# =========================================================

def plot_lawn_mower_trajectory(
    env: LawnMowerEnv,
    visit_counts=None,
    trajectory=None,
    agent_position=None,
    ax=None,
    title: str = "Lawn Mower Trajectory",
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
    else:
        fig = ax.figure

    ax.set_title(title)

    rows, cols = env.grid.shape

    # =====================================================
    # NORMALIZAÇÃO DO HEATMAP
    # =====================================================

    max_visits = 1

    if visit_counts is not None and len(visit_counts) > 0:
        max_visits = max(visit_counts.values())

    # =====================================================
    # GRID
    # =====================================================

    for r in range(rows):
        for c in range(cols):

            value = env.grid[r, c]

            # =============================================
            # OBSTÁCULOS
            # =============================================

            if value == 0:

                color = "black"

            else:

                # =========================================
                # HEATMAP
                # =========================================

                if visit_counts is not None:

                    visits = visit_counts.get((r,c), 0)

                    normalized = visits / max_visits

                    color = cm.RdYlGn_r(normalized)

                else:

                    if value == 2:
                        color = "green"

                    elif value == 3:
                        color = "red"

                    else:
                        color = "white"

            rect = plt.Rectangle(
                (c, r),
                1,
                1,
                facecolor=color,
                edgecolor="gray"
            )

            ax.add_patch(rect)

            # =============================================
            # VISIT COUNTS
            # =============================================

            if (
                visit_counts is not None
                and
                value != 0
            ):

                visits = visit_counts.get((r,c), 0)

                ax.text(
                    c + 0.82,
                    r + 0.22,
                    str(visits),
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                        pad=1
                    )
                )

    # =====================================================
    # TRAJECTORY
    # =====================================================

    if trajectory is not None and len(trajectory) > 1:

        xs = [c + 0.5 for (r,c) in trajectory]
        ys = [r + 0.5 for (r,c) in trajectory]

        ax.plot(
            xs,
            ys,
            linewidth=2,
            color="blue",
            alpha=0.8,
            zorder=10
        )

        # =============================================
        # START POINT
        # =============================================

        start_r, start_c = trajectory[0]

        ax.scatter(
            start_c + 0.5,
            start_r + 0.5,
            s=150,
            color="cyan",
            edgecolors="black",
            zorder=20,
            label="Start"
        )

        # =============================================
        # END POINT
        # =============================================

        end_r, end_c = trajectory[-1]

        ax.scatter(
            end_c + 0.5,
            end_r + 0.5,
            s=150,
            color="magenta",
            edgecolors="black",
            zorder=20,
            label="End"
        )

    # =====================================================
    # AGENT
    # =====================================================

    if agent_position is not None:

        r, c = agent_position

        circle = plt.Circle(
            (c + 0.5, r + 0.5),
            0.25,
            color="blue",
            zorder=30
        )

        ax.add_patch(circle)

    # =====================================================
    # AXIS
    # =====================================================

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    ax.set_xticks(np.arange(cols + 1))
    ax.set_yticks(np.arange(rows + 1))

    ax.grid(True)

    ax.invert_yaxis()

    ax.set_aspect("equal")

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.legend()

    return fig, ax

def plot_training_rewards(episode_rewards, output_dir, algorithm_name):
    fig_rewards, ax_rewards = (
        plt.subplots(
            figsize=(10, 5)
        )
    )

    ax_rewards.plot(
        episode_rewards,
        label="Episode Rewards",
    )

    ax_rewards.set_title(f"{algorithm_name} Training Rewards")

    ax_rewards.set_xlabel("Episode")

    ax_rewards.set_ylabel("Total Reward")

    ax_rewards.legend()

    ax_rewards.grid(True)

    plt.savefig(output_dir / "training_rewards.png", dpi=150, bbox_inches="tight")

    plt.show(block=False) 


def plot_episode_lengths(episode_lengths, output_dir, algorithm_name):
    fig_lengths, ax_lengths = (
        plt.subplots(
            figsize=(10, 5)
        )
    )

    ax_lengths.plot(
        episode_lengths,
        label="Episode Lengths",
    )

    ax_lengths.set_title(f"{algorithm_name} Episode Lengths")

    ax_lengths.set_xlabel("Episode")

    ax_lengths.set_ylabel("Steps")

    ax_lengths.legend()

    ax_lengths.grid(True)

    plt.savefig(output_dir / "episode_lengths.png", dpi=150, bbox_inches="tight")

    plt.show(block=False) 

def plot_coverage_heatmap(env, visit_counts, trajectory, position, output_dir, algorithm_name):
    fig_heatmap, ax_heatmap = (
        plot_lawn_mower_trajectory(
            env,
            visit_counts=visit_counts,
            trajectory=trajectory,
            agent_position=position,
            title=(
                f"{algorithm_name} Final Coverage "
                f"Heatmap + Trajectory"
            ),
        )
    )

    plt.savefig(output_dir / "coverage_heatmap.png", dpi=150, bbox_inches="tight")

    plt.show(block=False) 