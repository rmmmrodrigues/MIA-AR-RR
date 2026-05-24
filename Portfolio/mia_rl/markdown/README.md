# Portfólio de Rui Rodrigues pg7942
## MIA — Aprendizagem por Reforço 2026
### Universidade do Minho

---

# Visão Geral

Este repositório contém um conjunto de implementações e experiências desenvolvidas no contexto da unidade curricular de Aprendizagem por Reforço.

O objetivo principal do projeto foi estudar diferentes paradigmas de Reinforcement Learning através de:

- ambientes tabulares;
- métodos de Dynamic Programming;
- Temporal Difference Learning;
- Monte Carlo Methods;
- Function Approximation;
- Policy Gradient;
- Planning;
- Coverage Path Planning.

O projeto encontra-se organizado segundo uma arquitetura modular baseada na separação entre:

- ambientes;
- agentes;
- políticas;
- experiências;
- visualização;
- outputs experimentais.

---

# Execução

Toda a execução experimental deve ser realizada através de:

```bash
launcher.py
```

O launcher centraliza:

- inicialização dos ambientes;
- configuração dos algoritmos;
- execução de treino;
- geração de métricas;
- criação de plots;
- persistência dos resultados.

---

# Estrutura Técnica

```text
mia_rl/

    core/
    envs/
    agents/
    policies/
    mdps/
    experiments/
    plots/
    scripts/
    outputs/
```

---

# Descrição dos Componentes

## core/

Define as abstrações fundamentais:

- Agent
- Environment
- Policy
- Episode
- Transition

Permite desacoplamento entre:
- lógica de aprendizagem;
- dinâmica dos ambientes;
- estratégias de exploração.

---

## envs/

Implementação dos ambientes RL.

Inclui:
- GridWorld;
- Windy GridWorld;
- Blackjack;
- TicTacToe;
- K-Armed Bandits;
- Lawn Mower.

Cada ambiente define formalmente:

```text
(S, A, P, R)
```

onde:
- `S` representa estados;
- `A` representa ações;
- `P` representa transições;
- `R` representa recompensas.

---

## agents/

Implementação dos algoritmos de aprendizagem.

### Algoritmos implementados

- Monte Carlo
- SARSA
- N-Step SARSA
- Q-Learning
- TD Learning
- REINFORCE
- MCTS
- Linear Approximation
- Torch Approximation

---

## experiments/

Pipeline experimental responsável por:

- treino;
- avaliação;
- recolha de métricas;
- comparação entre algoritmos.

---

## plots/

Visualização dos resultados experimentais.

Inclui:
- learning curves;
- reward plots;
- heatmaps;
- trajectory visualization;
- convergence analysis.

---

# Organização Experimental

As experiências foram desenvolvidas para comparar:

- métodos on-policy;
- métodos off-policy;
- Monte Carlo vs TD;
- tabular vs approximation;
- planning vs learning.

Cada use case inclui:
- métricas;
- análise de convergência;
- avaliação da política;
- comparação entre algoritmos.

---

# Documentação Técnica

## Casos de Estudo

- Lawn_Mower.md
- Windy_GridWorld.md
- TicTacToe.md
- Blackjack.md
- KBandits.md

Cada documento descreve:
- arquitetura;
- representação do estado;
- reward shaping;
- algoritmos;
- métricas;
- resultados;
- comparação experimental.
