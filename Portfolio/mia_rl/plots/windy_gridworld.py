from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mia_rl.envs.windy_gridworld import WindyGridworldAction, WindyGridworldEnv, WindyGridworldState

ARROWS = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
}


def plot_episode_lengths(
    output_dir, 
    algorithm, 
    lengths: list[int],
    title: str = "Episode length over training",
):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(lengths)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")
    ax.set_title(f"{algorithm} {title}")
    plt.show(block=False)
    plt.savefig(output_dir / "lengths.png", dpi=150, bbox_inches="tight")


def plot_episode_rewards(
    output_dir, 
    algorithm, 
    rewards: list[float], 
    title: str = "Episode reward over training",
):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"{algorithm} {title}")
    plt.show(block=False)
    plt.savefig(output_dir / "rewards.png", dpi=150, bbox_inches="tight")    


def plot_policy(
    output_dir, 
    algorithm, 
    env: WindyGridworldEnv,
    policy: dict[WindyGridworldState, WindyGridworldAction],
    path: list[WindyGridworldState] | None = None,
    title: str = "Learned greedy policy",
):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title(f"{algorithm} {title}")
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(np.arange(env.cols + 1))
    ax.set_yticks(np.arange(env.rows + 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            wind_strength = env.wind[col]
            ax.text(col + 0.15, row + 0.2, str(wind_strength), fontsize=8, alpha=0.6)
            if state == env.goal:
                ax.text(col + 0.5, row + 0.55, "G", ha="center", va="center", fontsize=14)
                continue
            if state == env.start:
                ax.text(col + 0.25, row + 0.55, "S", ha="center", va="center", fontsize=14)
            action = policy.get(state)
            if action is not None:
                ax.text(col + 0.6, row + 0.55, ARROWS[action], ha="center", va="center", fontsize=16)

    if path is not None and len(path) > 1:
        xs = [col + 0.5 for _, col in path]
        ys = [row + 0.5 for row, _ in path]
        ax.plot(xs, ys, color="tab:red", linewidth=2, marker="o", markersize=4)

    plt.savefig(output_dir / "policy.png", dpi=150, bbox_inches="tight")   
    plt.show(block=False)



def plot_td_errors(
    output_dir, 
    algorithm, 
    errors: list[float],
    window: int = 20,
    title: str = "Mean |TD error| per episode",
):
    """Plot per-episode mean absolute TD error with a rolling-mean overlay."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    arr = np.array(errors)
    ax.plot(arr, alpha=0.3, color="tab:orange", label="per episode")
    if len(arr) >= window:
        rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(arr)), rolling, color="tab:orange", label=f"{window}-ep mean")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean |δ|")
    ax.set_title(f"{algorithm} {title}")
    
    plt.savefig(output_dir / "td_errors.png", dpi=150, bbox_inches="tight") 
    plt.show(block=False)



def plot_value_heatmap(
    output_dir, 
    algorithm, 
    env: WindyGridworldEnv,
    value_fn,
    title: str = "Learned state values V(s)",
):
    """Heatmap of V(s) = value_fn(s) for all grid states.

    value_fn can be:
      - LinearTD0:          lambda s: agent.value_of(s)
      - LinearSarsaControl: lambda s: max(agent.action_value_of(s, a) for a in ACTIONS)
    """
    grid = np.zeros((env.rows, env.cols))
    for row in range(env.rows):
        for col in range(env.cols):
            grid[row, col] = value_fn((row, col))

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(grid, origin="upper", aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="V(s)")
    ax.set_title(f"{algorithm} {title}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    start_row, start_col = env.start
    goal_row, goal_col = env.goal
    ax.text(start_col, start_row, "S", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.text(goal_col, goal_row, "G", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    
    plt.savefig(output_dir / "value_heatmap.png", dpi=150, bbox_inches="tight") 
    plt.show(block=False)
    


def plot_episode_length_comparison(
    output_dir, 
    algorithm, 
    lengths_dict: dict[str, list[int]],
    window: int = 20,
    title: str = "Episode length comparison",
):
    """Overlay episode-length curves for multiple agents with rolling-mean smoothing."""
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for label, lengths in lengths_dict.items():
        arr = np.array(lengths, dtype=float)
        ax.plot(arr, alpha=0.15)
        if len(arr) >= window:
            rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(arr)), rolling, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")
    ax.set_title(f"{algorithm} {title}")
    ax.legend()
    
    plt.savefig(output_dir / "value_heatmap.png", dpi=150, bbox_inches="tight") 
    plt.show(block=False)


def rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")

def plot_length_comparison(output_dir, algorithm, lengths_by_agent: dict[str, list[int]], window: int = 20):
    import matplotlib.pyplot as plt

    colors = {
        "Tabular SARSA": "#1b9e77",
        "Linear SARSA (NumPy)": "#d95f02",
        "Torch SARSA (manual)": "#7570b3",
        "Torch SARSA (optimizer)": "#e7298a",
    }
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for label, lengths in lengths_by_agent.items():
        arr = np.asarray(lengths, dtype=float)
        color = colors[label]
        ax.plot(arr, alpha=0.12, linewidth=1.0, color=color)
        if len(arr) >= window:
            smoothed = rolling_mean(arr, window)
            xs = np.arange(window - 1, len(arr))
        else:
            smoothed = arr
            xs = np.arange(len(arr))
        ax.plot(xs, smoothed, linewidth=2.4, color=color, label=label)
        ax.scatter(xs[-1], smoothed[-1], color=color, s=28, zorder=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")
    ax.set_title("Windy Gridworld: episode length comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    plt.savefig(output_dir / "comparison_lengths.png", dpi=150, bbox_inches="tight") 

    plt.show(block=False)    


def plot_td_error_panels(output_dir, algorithm, td_errors_by_agent: dict[str, list[float]], window: int = 20):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(td_errors_by_agent), figsize=(15, 4.2), constrained_layout=True, sharey=True)
    if len(td_errors_by_agent) == 1:
        axes = [axes]

    max_y = max(max(errors) for errors in td_errors_by_agent.values() if errors)
    for ax, (name, errors) in zip(axes, td_errors_by_agent.items()):
        arr = np.asarray(errors, dtype=float)
        ax.plot(arr, alpha=0.25, color="tab:orange", linewidth=1.0)
        if len(arr) >= window:
            smoothed = rolling_mean(arr, window)
            ax.plot(np.arange(window - 1, len(arr)), smoothed, color="tab:orange", linewidth=2.2)
        ax.set_title(name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean |δ|")
        ax.set_ylim(0.0, max_y * 1.05 if max_y > 0 else 1.0)
        ax.grid(alpha=0.25)
    fig.suptitle("TD error comparison")
    plt.savefig(output_dir / "td_errors.png", dpi=150, bbox_inches="tight") 

    plt.show(block=False)  

def draw_value_panel(ax, env, agent, title: str, actions, vmin: float, vmax: float):
    grid = np.zeros((env.rows, env.cols))
    for row in range(env.rows):
        for col in range(env.cols):
            grid[row, col] = max(agent.action_value_of((row, col), action) for action in actions)
    im = ax.imshow(grid, origin="upper", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    sr, sc = env.start
    gr, gc = env.goal
    ax.text(sc, sr, "S", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    return im


def draw_policy_panel(ax, env, policy, path, title: str):
    arrows = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    ax.set_title(title)
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(np.arange(env.cols + 1))
    ax.set_yticks(np.arange(env.rows + 1))
    ax.grid(True, alpha=0.35)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            ax.text(col + 0.14, row + 0.18, str(env.wind[col]), fontsize=7, alpha=0.55)
            if state == env.goal:
                ax.text(col + 0.5, row + 0.55, "G", ha="center", va="center", fontsize=13)
                continue
            if state == env.start:
                ax.text(col + 0.25, row + 0.55, "S", ha="center", va="center", fontsize=13)
            action = policy.get(state)
            if action is not None:
                ax.text(col + 0.62, row + 0.55, arrows[action], ha="center", va="center", fontsize=14)

    if path is not None and len(path) > 1:
        xs = [col + 0.5 for _, col in path]
        ys = [row + 0.5 for row, _ in path]
        ax.plot(xs, ys, color="tab:red", linewidth=1.8, marker="o", markersize=3.5)

def plot_value_heatmaps(
    output_dir, 
    algorithm, 
    agents, 
    env, 
    ACTIONS, 
    vmin, vmax
):
    fig_heat, axes_heat = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    last_im = None
    for ax, (name, agent) in zip(axes_heat.ravel(), agents.items()):
        last_im = draw_value_panel(ax, env, agent, name, ACTIONS, vmin=vmin, vmax=vmax)
    if last_im is not None:
        fig_heat.colorbar(last_im, ax=axes_heat.ravel().tolist(), shrink=0.92, label="V(s) = max_a q(s, a)")
    fig_heat.suptitle("Windy Gridworld: learned value surfaces")

    plt.savefig(output_dir / "value_heatmap.png", dpi=150, bbox_inches="tight") 

    plt.show(block=False)    

def plot_policy_comparison(
    output_dir, 
    algorithm, 
    policies, 
    env, 
    paths
):        
    fig_policy, axes_policy = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (name, policy) in zip(axes_policy.ravel(), policies.items()):
        draw_policy_panel(ax, env, policy, paths[name], f"{name} | path={len(paths[name]) - 1}")
    fig_policy.suptitle("Windy Gridworld: greedy policy comparison")

    plt.savefig(output_dir / "policy_comparison.png", dpi=150, bbox_inches="tight") 

    plt.show(block=False)      