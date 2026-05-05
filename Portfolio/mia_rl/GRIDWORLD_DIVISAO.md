# Divisao do MDP GridWorld

Este ficheiro documenta a divisao do notebook `notebooks/MDP_GridWorld.ipynb` pela arquitetura do pacote `mia_rl`: o notebook passa a importar codigo do pacote em vez de concentrar todas as classes e funcoes nas celulas.

## Estrutura criada

```text
mia_rl/

  agents/
    planning/
      gridworld.py - reutilizado e adicionadas funções

  envs/
    gridworld.py - reutilizado

  experiments/
    gridworld.py

  policies/
    gridworld.py - reutilizado e adicionadas funções

  plots/
    mdp_gridworld.py - reutilizado e adicionadas funções

  scripts/
    run_gridworld.py
    run_gridworld.bat
```

## Divisao dos blocos

- `mia_rl/agents/planning/gridworld.py`
  - `policy_iteration`

- `mia_rl/experiments/gridworld.py`  
  - corre os exemplos principais do notebook
  - grava os graficos em `mia_rl/outputs/gridworld`

- `mia_rl/policies/gridworld.py`
  - `greedy_action_from_V`
  - `policy_improvement`

- `mia_rl/plots/mdp_gridworld.py`   
  - `plot_grid`

- `mia_rl/scripts/run_gridworld.py`
  - mia_rl/experiments/gridworld.py

## Como executar

Em PowerShell:

```powershell
& .\MIA-AR-RR\Portfolio\mia_rl\scripts\run_gridworld.bat
```

Ou diretamente com Python:

```powershell
python MIA-AR-RR\Portfolio\mia_rl\scripts\run_gridworld.py
```
