# Divisao do MDP GridWorld

Este ficheiro documenta a divisao do notebook `notebooks/MDP_GridWorld.ipynb` pela arquitetura do pacote `mia_rl`, seguindo o mesmo principio usado em `TicTacToe_MCTS.ipynb`: o notebook passa a importar codigo do pacote em vez de concentrar todas as classes e funcoes nas celulas.

## Estrutura criada

```text
mia_rl/

  agents/
    planning/
      gridworld.py

  envs/
    gridworld.py

  experiments/
    gridworld.py

  policies/
    gridworld.py

  plots/
    mdp_gridworld.py

  scripts/
    run_mdp_gridworld.py
    run_mdp_gridworld.bat
```

## Divisao dos blocos

- `mia_rl/agents/planning/gridworld.py`
  - `zeros_V`
  - `zeros_Q`
  - `bellman_expectation_update`
  - `policy_evaluation`
  - `policy_evaluation_with_history`
  - `bellman_optimality_update`
  - `value_iteration`
  - `expected_backup_optimal_stochastic`
  - `value_iteration_stochastic`
  - `policy_evaluation_Q`  

- `mia_rl/envs/gridworld.py`
  - `Gridworld`
  - `TrapGridworld`
  - `ACTIONS`
  - `ACTION_TO_DELTA`

- `mia_rl/experiments/mdp_gridworld.py`  
  - corre os exemplos principais do notebook
  - grava os graficos em `mia_rl/outputs/mdp_gridworld`

- `mia_rl/policies/gridworld.py`
  - `uniform_random_policy`
  - `greedy_policy_from_V`

- `mia_rl/plots/mdp_gridworld.py`
  - `plot_grid_values_and_policy`

- `mia_rl/scripts/run_mdp_gridworld.py`
  - mia_rl/experiments/gridworld.py

## Como executar

Em PowerShell:

```powershell
& .\MIA-AR-RR\Portfolio\mia_rl\scripts\run_mdp_gridworld.bat
```

Ou diretamente com Python:

```powershell
python MIA-AR-RR\Portfolio\mia_rl\scripts\run_mdp_gridworld.py
```
