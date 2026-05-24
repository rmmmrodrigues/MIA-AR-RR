
import matplotlib.pyplot as plt

from mia_rl.plots.mdp_gridworld import plot_grid


def plot_policy_evaluation(output_dir, env, V_pi0):
    fig, _ = plot_grid(env, V_pi0, policy=None, title="Policy evaluation: V^π (uniform random π)")
    fig.savefig(output_dir / "policy_evaluation.png")
    plt.show(block=False)

def plot_policy_improvement(output_dir, env, V_pi0, pi1_actions):
    fig, _ = plot_grid(env, V_pi0, policy=pi1_actions, title="Greedy policy w.r.t. V^π (arrows)")
    fig.savefig(output_dir / "policy_improvement.png")
    plt.show(block=False)  

def plot_policy_iteration(output_dir, env, V_star, pi_star_actions):
    fig, _ = plot_grid(env, V_star, policy=pi_star_actions, title="Policy Iteration: V* and π* (greedy actions)")
    fig.savefig(output_dir / "policy_iteration.png")
    plt.show(block=False)  

def plot_policy_iteration_all(output_dir, num_plots, env, hist):    
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 6, 6)) # Adjust figsize for horizontal plots

    # Ensure axes is an array even for a single plot
    if num_plots == 1:
        axes = [axes]

    for i, (outer_iter, pe_iters, V_hist, pi_actions_hist) in enumerate(hist):
        ax = axes[i]
        fig, _ = plot_grid(env, V_hist, policy=pi_actions_hist, title=f"Policy Iteration (Outer Loop) {outer_iter}\nPolicy Evaluation (Inner Loop) Itrs: {pe_iters}", ax=ax)
        
    fig.savefig(output_dir / "policy_iteration_all.png")
    
    plt.show(block=False)  