from __future__ import annotations

from typing import Callable

from mia_rl.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState, _winner

# Policy type: a callable that takes (env, state) and returns an action.
Policy = Callable[[TicTacToeEnv, TicTacToeState], TicTacToeAction]


def play_game(
    env: TicTacToeEnv,
    policy_x: Policy,
    policy_o: Policy,
    render: bool = True,
) -> int:
    """Play one full game between two policies, optionally rendering each step.

    Args:
        env: the TicTacToeEnv instance.
        policy_x: callable (env, state) -> action for player X (+1).
        policy_o: callable (env, state) -> action for player O (-1).
        render: if True, print the board after every move.

    Returns:
        1 if X wins, -1 if O wins, 0 for a draw.
    """
    state = env.reset()
    if render:
        print("Initial board:")
        env.render(state)
        print()

    while not env.is_terminal(state):
        player_label = "X" if env.current_player == 1 else "O"

        # TODO: select the correct policy for the current player and get an action.
        # Hint: consider what env.current_player encodes and how policy_x / policy_o
        # relate to it — the same attribute you used in encode_state.
        #raise NotImplementedError("TODO: select policy and get action.")
        
        if env.current_player == 1:
            action = policy_x(env, state)
        else:
            action = policy_o(env, state)

        state, reward, done = env.step(action)

        if render:
            print(f"Player {player_label} plays cell {action}:")
            env.render(state)
            print()

    result = _winner(state)
    if render:
        if result == 1:
            print("X wins!")
        elif result == -1:
            print("O wins!")
        else:
            print("Draw!")
    return result
