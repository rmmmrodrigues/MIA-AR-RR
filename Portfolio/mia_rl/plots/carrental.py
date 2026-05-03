from typing import Dict, Tuple
from matplotlib import pyplot as plt
import numpy as np
from mia_rl.envs.carrental import CarRentalMDP

def policy_to_array(mdp: CarRentalMDP, policy: Dict[Tuple[int,int], int]) -> np.ndarray:
    arr = np.zeros((mdp.params.max_cars_1 + 1, mdp.params.max_cars_2 + 1), dtype=int)
    for (n1, n2), a in policy.items():
        arr[n1, n2] = a
    return arr

def plot_policy(mdp: CarRentalMDP, policy: Dict[Tuple[int,int], int], title: str = ""):
    arr = policy_to_array(mdp, policy)

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(arr, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("# cars at location 2")
    ax.set_ylabel("# cars at location 1")

    # annotate
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i,j]), ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.show()

    return fig

def plot_values(mdp: CarRentalMDP, V: np.ndarray, title: str = ""):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(V, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("# cars at location 2")
    ax.set_ylabel("# cars at location 1")
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.show()

    return fig