import random
from mia_rl.envs.tictactoe import TicTacToeEnv

env = TicTacToeEnv()
state = env.reset()

while True:
    env.render()
    print()

    if env.current_player == 1:
        action = int(input("Your move (0-8): "))
    else:
        action = random.choice(env.available_actions(state))
        print(f"Random plays: {action}")

    try:
        state, reward, done = env.step(action)
    except ValueError as e:
        print(e)
        continue

    if done:
        env.render()
        print("Game over!")
        break
