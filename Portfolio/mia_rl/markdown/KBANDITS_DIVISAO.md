# Divisao do k-armed bandit

Este ficheiro documenta a divisao simples do exemplo `kbandits` pela arquitetura do pacote `mia_rl`.

Foram mantidos apenas os blocos existentes no exemplo original:

- ambiente
- agentes
- runner da experiencia
- funcoes de plot
- main

## Estrutura criada

```text
mia_rl/
  envs/
    kbandits.py

  agents/
    control/
      kbandits.py

  experiments/
    kbandits.py

  plots/
    kbandits.py

  scripts/
    run_kbandits.py
    run_kbandits.bat
```

## Divisao dos blocos

- `mia_rl/envs/kbandits.py`
  - `KArmedBandit`

- `mia_rl/agents/control/kbandits.py`
  - `BanditAgent`
  - `EpsilonGreedy`
  - `UCB`
  - `GradientBandit`

- `mia_rl/experiments/kbandits.py`
  - `run_experiment`

- `mia_rl/plots/kbandits.py`
  - `plot_epsilon_greedy`
  - `plot_optimistic_vs_ucb`
  - `plot_gradient_bandit`

- `mia_rl/scripts/run_kbandits.py`
  - `main`
  - cria uma figura para cada plot
  - guarda os graficos em `mia_rl/outputs/kbandits`
  - apresenta os graficos no ecra com `plt.show()`

## Como executar

Em PowerShell:

```powershell
& .\MIA-AR-RR\Portfolio\mia_rl\scripts\run_kbandits.bat
```

Ou diretamente com Python:

```powershell
python MIA-AR-RR\Portfolio\mia_rl\scripts\run_kbandits.py
```
