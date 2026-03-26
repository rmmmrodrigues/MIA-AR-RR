from mia_rl.envs.tictactoe import TicTacToeEnv

env = TicTacToeEnv()
state = env.reset()

while True:
    env.render()
    print()

    player = "X" if env.current_player == 1 else "O"
    action = int(input(f"Player {player}, choose position (0-8): "))

    try:
        state, reward, done = env.step(action)
    except ValueError as e:
        print(e)
        continue

    if done:
        env.render()
        print()

        if reward == 1:
            print(f"Player {player} wins!")
        elif reward == -1:
            print(f"Player {player} loses!")
        else:
            print("Draw!")

        break
