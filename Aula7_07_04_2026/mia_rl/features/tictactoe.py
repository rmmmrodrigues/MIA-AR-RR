from __future__ import annotations

import random

import numpy as np

from mia_rl.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState

# Each of the 9 cells is encoded as a 3-dim one-hot vector (from the current
# player's perspective): [my piece, opponent's piece, empty].
# Total feature dimension: 9 × 3 = 27.
STATE_FEATURE_DIM: int = 27


def encode_state(board: TicTacToeState, current_player: int) -> np.ndarray:
    """Encode a board as a 27-dim one-hot vector from `current_player`'s perspective.

    For each cell the encoding is:
        [1, 0, 0]  if the cell contains current_player's mark
        [0, 1, 0]  if the cell contains the opponent's mark
        [0, 0, 1]  if the cell is empty

    Using a perspective-relative encoding means the same policy weights work
    regardless of whether the agent is playing as X (+1) or O (-1).

    Args:
        board: length-9 tuple of ints (0 = empty, 1 = X, -1 = O).
        current_player: +1 or -1, identifies whose turn it is.

    Returns:
        np.ndarray of shape (27,), dtype float32.

    TODO:
    1. Create a zero array of shape (STATE_FEATURE_DIM,) with dtype float32.
    2. Loop over each cell index i and its value in `board`.
    3. For each cell set exactly one of the three slots to 1.0:
       - slot i*3 + 0  if cell == current_player  (my piece)
       - slot i*3 + 1  if cell == -current_player (opponent's piece)
       - slot i*3 + 2  if cell == 0               (empty)
    4. Return the feature vector.
    """
    #raise NotImplementedError("TODO: implement encode_state.")

    # 1. Create a zero array of shape (STATE_FEATURE_DIM,) with dtype float32.
    features = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)

    # 2.Loop over each cell index i and its value in `board`.
    for i, cell in enumerate(board):
        base_idx = i * 3

        # 3. For each cell set exactly one of the three slots to 1.0:
        if cell == current_player:
            features[base_idx + 0] = 1.0
        elif cell == -current_player:
            features[base_idx + 1] = 1.0
        else:  # cell == 0
            features[base_idx + 2] = 1.0

    # 4. Return feature vector
    return features


def random_action(env: TicTacToeEnv, state: TicTacToeState) -> TicTacToeAction:
    """Choose a uniformly random legal action. Used as a baseline opponent."""
    return random.choice(env.available_actions(state))
