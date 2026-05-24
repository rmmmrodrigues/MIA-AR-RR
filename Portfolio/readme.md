# Portfólio de Rui Rodrigues — pg7942
## MIA — Aprendizagem por Reforço 2026
### Universidade do Minho

---

## Índice

1. [Objetivo do Repositório](#1-objetivo-do-repositório)
2. [Instalação e Execução](#2-instalação-e-execução)
3. [Arquitetura do Package](#3-arquitetura-do-package)
4. [Algoritmos Implementados](#4-algoritmos-implementados)
5. [Experiências por Ambiente](#5-experiências-por-ambiente)
6. [Comparação de Algoritmos por Use Case](#6-comparação-de-algoritmos-por-use-case)

---

## 1. Objetivo do Repositório

Este repositório contém a implementação prática dos principais algoritmos estudados na unidade curricular de **Aprendizagem por Reforço (Reinforcement Learning)** da licenciatura/mestrado MIA.

O projeto cobre, de forma progressiva, os paradigmas centrais de RL:

- **Processos de Decisão de Markov** com modelo conhecido (Programação Dinâmica)
- **Métodos tabulares model-free** (Monte Carlo, TD, SARSA, Q-Learning)
- **Métodos multi-step** (N-Step SARSA, TD(n))
- **Aproximação de função linear** (Linear SARSA, Linear TD, PyTorch SARSA)
- **Policy Gradient** (REINFORCE)
- **Planeamento por simulação** (Monte Carlo Tree Search)
- **Bandit algorithms** (ε-greedy, UCB, Gradient Bandit)

---

## 2. Instalação e Execução

### Ambiente Conda

```bash
conda env create -f mia_rl/environment.yml
conda activate mia_rl
```

### Lançar Experiências

```bash
python mia_rl/launcher.py
```

O `launcher.py` é o ponto de entrada central. Ao ser executado, apresenta um menu interativo numerado com todos os scripts em `mia_rl/scripts/run_*.py`. O utilizador seleciona o número da experiência desejada, o script é lançado como subprocesso, e ao terminar é possível regressar ao menu ou sair.

Exemplo de interação:

```
===================================
 RL Experiment Launcher
===================================

1. Bandits
2. Blackjack Prediction
3. Carrental
4. Gridworld
...

0. Exit

Select experiment: 2
```

Cada script de experiência pode também aceitar argumentos de linha de comandos, documentados com `--help`.

---

## 3. Arquitetura do Package

```
mia_rl/
├── core/           # Abstrações base (agentes, políticas, episódios, transições)
├── envs/           # Ambientes de interação
├── mdps/           # Representações explícitas de MDPs
├── agents/
│   ├── control/    # Algoritmos de controlo (SARSA, Q-Learning, MC, REINFORCE, ...)
│   ├── prediction/ # Algoritmos de predição (MC, TD(0), TD(n), Linear TD)
│   └── planning/   # Programação dinâmica e MCTS
├── policies/       # Estratégias de seleção de ações
├── features/       # Vetores de features para aproximação de função
├── experiments/    # Lógica de treino, geração de episódios, avaliação
├── plots/          # Visualização de resultados
├── scripts/        # Scripts run_*.py para cada experiência
├── outputs/        # Resultados, gráficos e logs gerados
└── launcher.py     # Ponto de entrada central
```

### `core/`

Define as abstrações fundamentais do framework:

- `Transition` — tuplo `(state, action, reward, next_state, done)` que representa uma transição de ambiente.
- `Episode` — coleção ordenada de transições de um episódio completo.
- `PredictionAgent` — interface base para agentes de predição (expõe `value_of`, `update_episode`, `reset`).
- `ControlAgent` — interface base para agentes de controlo (expõe `select_action`, `update_transition`, `greedy_action`, `end_episode`).
- `Policy` — protocolo genérico para políticas.

Estas abstrações garantem o desacoplamento total entre ambiente e agente, permitindo substituir qualquer componente sem alterar os restantes.

### `envs/`

Cada ambiente implementa formalmente o espaço de estados, o espaço de ações, a dinâmica de transição `P(s'|s,a)` e a função de recompensa `R(s,a)`.

| Ambiente | Ficheiro | Descrição |
|---|---|---|
| K-Armed Bandits | `kbandits.py` | Problema de exploração/exploração estacionário |
| GridWorld | `gridworld.py` | Grelha determinística e estocástica com terminais |
| Windy GridWorld | `windy_gridworld.py` | Grelha com vento por coluna |
| Blackjack | `blackjack.py` | Jogo de cartas episódico |
| Car Rental | `carrental.py` | MDP de gestão de inventário de dois locais |
| Lawn Mower | `lawn_mower.py` | Coverage path planning em mapa com obstáculos |
| TicTacToe | `tictactoe.py` | Jogo adversarial de dois jogadores |

### `agents/control/`

Algoritmos de controlo on-policy e off-policy. Todos herdam de `ControlAgent` e partilham a mesma interface de interação com o ambiente.

### `agents/prediction/`

Algoritmos de predição (estimação de `V(s)` sob uma política fixa). Herdam de `PredictionAgent`.

### `agents/planning/`

Algoritmos que assumem conhecimento do modelo (DP) ou utilizam o modelo como simulador (MCTS).

### `features/`

Vetores de features para aproximação de função linear:

- `windy_gridworld.py` — tile coding para o par (estado, ação): codificação por blocos de ação independentes com one-hot da posição.
- `tictactoe.py` — encoding perspetivo-relativo em 27 dimensões (9 células × 3 categorias: minha peça / adversário / vazia).

---

## 4. Algoritmos Implementados

---

### K-Armed Bandits — `agents/control/kbandits.py`

Três agentes para o problema de bandits estacionários, todos herdam de `BanditAgent` (interface `select_action`, `update`):

#### ε-greedy (`EpsilonGreedy`)

Com probabilidade ε seleciona uma ação aleatória (exploração); caso contrário seleciona `argmax Q`. Suporta dois modos de atualização:

- **Média incremental** (`alpha=None`): `Q(a) += (R - Q(a)) / N(a)` — adequado para problemas estacionários.
- **Passo constante** (`alpha=float`): `Q(a) += alpha * (R - Q(a))` — adequado para problemas não-estacionários, onde recompensas recentes devem ter mais peso.

Suporta ainda **initialização otimista** (`optimistic > 0`): inicializar Q com valores altos força exploração inicial mesmo com ε=0.

#### UCB (`UCB`)

Seleciona a ação com maior *upper confidence bound*:

```
UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
```

O segundo termo penaliza ações pouco visitadas, garantindo que nenhuma ação é ignorada indefinidamente. O parâmetro `c` controla a largura do intervalo de confiança.

#### Gradient Bandit (`GradientBandit`)

Mantém preferências `H(a)` e seleciona ações por softmax: `π(a) = exp(H(a)) / Σ exp(H(b))`. Atualiza as preferências por gradiente:

```
H(a_t) += alpha * (R_t - baseline) * (1 - π(a_t))
H(a)   -= alpha * (R_t - baseline) * π(a)   ∀ a ≠ a_t
```

Com `baseline=True` usa a média incremental das recompensas como baseline, reduzindo a variância.

---

### Monte Carlo Prediction — `agents/prediction/monte_carlo.py`

**First-Visit Monte Carlo** para estimação de `V(s)` sob uma política fixa.

Processa episódios completos: percorre as transições de trás para a frente a computar os retornos descontados `G_t = r_t + γ * G_{t+1}`. Apenas atualiza `V(s)` na *primeira visita* a cada estado no episódio, usando média incremental:

```
N(s) += 1
V(s) += (G - V(s)) / N(s)
```

Sem bootstrapping — a estimativa de cada estado depende apenas dos retornos observados, nunca de estimativas de outros estados.

---

### TD(0) e TD(n) Prediction — `agents/prediction/td.py`

**TD(0)** (`TD0Prediction`) — atualização online por transição, com bootstrapping:

```
V(s) += alpha * (r + gamma * V(s') - V(s))
```

Processa cada transição individualmente sem aguardar o fim do episódio.

**TD(n)** (`TDNPrediction`) — retorno parcial de `n` passos:

```
G_t^(n) = r_t + γ*r_{t+1} + ... + γ^(n-1)*r_{t+n-1} + γ^n * V(s_{t+n})
```

Processa episódios completos mas usa uma janela de `n` passos em vez do retorno total. Ao aumentar `n`, TD(n) aproxima-se de Monte Carlo; com `n=1` é equivalente a TD(0).

---

### Monte Carlo Control — `agents/control/monte_carlo.py`

**First-Visit Monte Carlo Control** com ε-greedy. Acumula transições durante o episódio e, quando termina (ou é truncado), executa a atualização em `_update_from_episode`:

1. Computa os retornos descontados em passo inverso.
2. Para cada par (s,a) na sua *primeira visita*, atualiza `Q(s,a)` por média incremental.

A política é implícita (ε-greedy sobre Q); o método `greedy_action` devolve o argmax de Q para uso em avaliação.

O método `end_episode` permite fechar episódios truncados por `max_steps`, garantindo que as transições acumuladas não são perdidas.

---

### SARSA — `agents/control/sarsa.py`

**TD on-policy** com ε-greedy. Atualização por transição:

```
Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)]
```

O par `(s',a')` é a ação *efetivamente selecionada* no estado seguinte pela política atual — armazenada em `_selected_actions`. Isto torna SARSA on-policy: a atualização depende da política de comportamento, incluindo as explorações.

---

### Q-Learning — `agents/control/q_learning.py`

**TD off-policy** com ε-greedy decay. Atualização por transição:

```
Q(s,a) += alpha * [r + gamma * max_{a'} Q(s',a') - Q(s,a)]
```

O bootstrap usa `max Q(s',a')` em vez da ação selecionada — independentemente da política de comportamento. Converge para `Q*` mesmo que o agente explore agressivamente.

Inclui **epsilon decay**: `epsilon = max(epsilon_min, epsilon * epsilon_decay)` ao fim de cada episódio, reduzindo progressivamente a exploração.

---

### N-Step SARSA — `agents/control/n_step_sarsa.py`

Generalização multi-step de SARSA. Mantém um buffer circular de transições. Ao acumular `n` transições, computa o retorno:

```
G = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1} + gamma^n * Q(s_{t+n}, a_{t+n})
```

e atualiza `Q(s_t, a_t)`. No fim do episódio, `end_episode` drena o buffer sem bootstrap (`use_bootstrap=False`), garantindo que as transições restantes são processadas.

Valores maiores de `n` propagam recompensas mais rapidamente a estados distantes, ao custo de maior variância.

---

### Linear TD(0) — `agents/prediction/linear_td.py`

**Semi-gradient TD(0)** com aproximação linear de `v_hat(s) = w · φ(s)`. Atualização online por transição:

```
delta = r + gamma * (w · phi(s')) - (w · phi(s))
w    += alpha * delta * phi(s)
```

O gradiente é calculado apenas em relação ao estado atual `s`; o bootstrap `v_hat(s')` é tratado como constante (semi-gradiente). Usado para estimar `V^π(s)` em Windy GridWorld com tile coding.

---

### Linear SARSA — `agents/control/linear_sarsa.py`

**Semi-gradient SARSA** com aproximação linear de `q_hat(s,a) = w · φ(s,a)`. Usa um encoding por blocos de ação: o vetor de features de `(s,a)` tem zeros em todos os blocos exceto no bloco correspondente à ação `a`, garantindo pesos independentes por ação. Atualização:

```
delta = r + gamma * (w · phi(s',a')) - (w · phi(s,a))
w    += alpha * delta * phi(s,a)
```

Regista os TD errors em `_td_errors` para análise de convergência; acedidos via `flush_td_errors()`.

---

### Torch SARSA — `agents/control/torch_sarsa.py`

Implementação de SARSA linear em **PyTorch**, matematicamente idêntica a `LinearSarsaControl`. O objetivo é demonstrar duas formas de calcular a mesma atualização com autograd:

**Modo manual** (`use_optimizer=False`):
```python
loss = 0.5 * F.mse_loss(pred, target_tensor)
loss.backward()
model.weight.data -= alpha * model.weight.grad
```

**Modo optimizer** (`use_optimizer=True`):
```python
loss = 0.5 * F.mse_loss(pred, target_tensor.detach())
loss.backward()
optimizer.step()   # SGD com lr=alpha
```

O target é **sempre `detach()`-ado** — condição essencial para o semi-gradiente. O fator `0.5` garante que `grad_w = (pred - target) * phi`, tornando a atualização com `SGD(lr=alpha)` idêntica à do NumPy.

Inclui cache de tensores para evitar conversões repetidas e um passo batched para avaliar Q de todas as ações de uma vez.

---

### REINFORCE — `agents/control/reinforce.py`

**Monte Carlo Policy Gradient** para TicTacToe. A política é um softmax linear sobre ações disponíveis:

```
h(s,a)  = theta[a] · phi(s)
pi(a|s) = softmax_disponíveis(h)
```

com `phi(s)` o vetor de 27 features perspetivo-relativo. O mesmo `theta` serve para X e O em self-play.

Atualização no fim de cada episódio:

```
theta[a_t] += alpha * gamma^t * G_t * (1 - pi(a_t|s_t)) * phi(s_t)
theta[a]   -= alpha * gamma^t * G_t * pi(a|s_t) * phi(s_t)   ∀ a ≠ a_t
```

Suporta **regularização por entropia** (`entropy_beta > 0`), que penaliza políticas determinísticas e incentiva exploração contínua.

Em self-play, as trajetórias de X e de O são recolhidas separadamente; a recompensa final do perdedor é injetada manualmente como `-1` (o ambiente só emite `+1` para o vencedor).

---

### MCTS — `agents/planning/mcts.py`

**Monte Carlo Tree Search** para TicTacToe. Sem pesos aprendidos; usa o ambiente como modelo perfeito.

Cada nó `MCTSNode` representa um estado e mantém: `visit_count`, `value_sum` (da perspetiva do jogador nesse nó), ações não expandidas e filhos.

Score UCB1 (avaliado pelo pai, que é o adversário):

```
UCB(filho) = -Q(filho) + c * sqrt(ln(N_pai) / N_filho)
```

O sinal negativo reflete que o que é bom para o filho é mau para o pai.

Quatro fases por simulação:

1. **Selection** — desce pela árvore com UCB1 até encontrar nó não totalmente expandido.
2. **Expansion** — adiciona um filho para uma ação não tentada.
3. **Simulation** — rollout aleatório até estado terminal.
4. **Backup** — propaga o resultado pela árvore, alternando sinal em cada nível.

A ação escolhida é a do filho mais visitado (*robust best*), menos sensível a outliers do que a de maior Q médio.

---

### Dynamic Programming — `agents/planning/gridworld.py`, `agents/planning/carrental.py`

Algoritmos com conhecimento completo do modelo:

**Policy Evaluation**: resolve iterativamente a equação de Bellman para uma política π:
```
V(s) = Σ_a π(a|s) [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]
```
Converge quando `max |V_novo - V_antigo| < θ`.

**Policy Improvement**: para cada estado, toma a ação greedy em relação a V:
```
π'(s) = argmax_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]
```

**Policy Iteration**: alterna entre Policy Evaluation e Policy Improvement até estabilidade.

**Value Iteration**: combina os dois passos numa única operação (Bellman optimality backup):
```
V(s) = max_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]
```

Para Car Rental, `q_from_v` calcula o retorno esperado usando a distribuição de Poisson sobre chegadas e devoluções de carros nos dois locais, com custo linear de transporte entre eles.

---

## 5. Experiências por Ambiente

---

### K-Armed Bandits — `experiments/kbandits.py`

**Script:** `scripts/run_kbandits.py`

Compara os três agentes bandit em 2000 runs independentes de 1000 passos. Cada run reinicia o ambiente (novo vetor de recompensas verdadeiras) e o agente. Gera três grupos de plots em `outputs/kbandits/`:

- `epsilon_greedy`: ε ∈ {0, 0.01, 0.1} — curvas de recompensa média e percentagem de seleção da ação ótima.
- `optimistic_vs_ucb`: inicialização otimista Q=5 vs UCB c=2 com ε=0 — demonstra que a exploração forçada por inicialização otimista iguala UCB no curto prazo.
- `gradient_bandit`: com e sem baseline de recompensa — demonstra a importância do baseline para estabilidade.

---

### Blackjack Prediction — `experiments/blackjack_prediction.py`

**Script:** `scripts/run_blackjack_prediction.py`  
**Argumentos:** `--episodes 20000 --td-alpha 0.05 --threshold 20 --seed 7`

Compara três métodos de predição sob a política threshold (hit se soma < threshold):

- **First-Visit MC** — retornos completos, sem bootstrapping.
- **TD(0)** — atualização online por transição, `alpha=0.05`.
- **TD(n)** — n=5 passos, `alpha=0.05`.

Os três agentes são treinados com o mesmo número de episódios e avaliados nos estados `(player_sum, dealer_showing, usable_ace)`. São registados snapshots em checkpoints {1000, 5000, 20000} para visualizar a evolução temporal da estimativa de V(s).

Gráficos gerados em `outputs/blackjack_prediction/`:
- Superfícies de valor 3D para cada agente em cada checkpoint.
- Diferenças de valor entre agentes (comparação direta).

---

### GridWorld MDP — `experiments/mdp_gridworld.py` e `experiments/gridworld.py`

**Scripts:** `scripts/run_mdp_gridworld.py`, `scripts/run_gridworld.py`

Demonstração completa de Programação Dinâmica em dois ambientes:

- **GridWorld determinístico**: grelha 4×4 com terminais no canto. Corre Policy Evaluation com política uniforme aleatória, seguido de Policy Improvement e Policy Iteration completo. Plota o histórico de convergência da função de valor em cada iteração exterior.
- **GridWorld estocástico (TrapGridworld)**: dinâmica com probabilidade de desvio lateral. Compara Value Iteration determinístico vs estocástico.
- **MDP GridWorld com Q**: adiciona avaliação da Q-function sob a política uniforme.

Gráficos gerados em `outputs/mdp_gridworld/` e `outputs/gridworld/`:
- Heatmaps de V(s) por iteração.
- Setas de política (direção greedy por célula).

---

### Car Rental — `experiments/carrental.py`

**Script:** `scripts/run_carrental.py`

Problema clássico de Sutton & Barto. Dois locais de aluguer de carros; chegadas e devoluções seguem distribuições de Poisson. A ação é o número de carros movidos entre locais de noite (custo por carro movido; capacidade máxima por local).

Corre e compara:

- **Policy Iteration** — avaliação + melhoria iterativa até convergência.
- **Value Iteration** — backup de otimalidade direto.

Gera heatmaps da política ótima (quantidade de carros a mover por estado `(n1, n2)`) e da função de valor, em `outputs/carrental/`.

---

### Windy GridWorld (controlo tabular) — vários scripts

**Scripts:** `run_windy_gridworld_sarsa.py`, `run_windy_gridworld_n_step_sarsa.py`, `run_windy_gridworld_mc_control.py`

Ambiente com grelha 7×10, vento variável por coluna e objetivo fixo. O agente aprende uma política de navegação from-scratch sem modelo.

Cada script treina um agente independente e gera em `outputs/windy_gridworld_*/`:
- Curva de comprimento de episódio ao longo do treino.
- Curva de recompensa por episódio.
- Visualização da política greedy aprendida (seta por célula).
- Caminho ótimo do ponto de início ao objetivo.

Os três scripts são estruturalmente idênticos na lógica de treino — a única diferença é o agente instanciado, o que facilita a comparação direta entre algoritmos.

---

### Windy GridWorld (aproximação de função) — vários scripts

**Scripts:** `run_windy_gridworld_linear_td.py`, `run_windy_gridworld_linear_sarsa.py`, `run_windy_gridworld_torch_sarsa.py`

Extensão da experiência tabular para aproximação de função linear com tile coding.

**`windy_gridworld_linear_td.py`**: pipeline em dois passos:
1. Treina um `SarsaControl` tabular para 5000 episódios como política de comportamento.
2. Usa essa política fixa para treinar `LinearTD0` (predição), estimando `V^π(s) = w · φ(s)`.

Gera heatmap de valor e curva de convergência dos TD errors em `outputs/windy_gridworld_linear_td/`.

**`windy_gridworld_linear_sarsa.py`**: controlo direto com `LinearSarsaControl`, sem pré-treino. Gera além das curvas de treino um heatmap do valor da política aprendida.

**`windy_gridworld_torch_sarsa.py`**: script de comparação quádrupla que treina os quatro agentes — Tabular SARSA, Linear SARSA (NumPy), Torch SARSA (manual), Torch SARSA (optimizer) — com os mesmos hiperparâmetros e sobrepõe as curvas de comprimento de episódio e heatmaps de valor. É o script principal para demonstrar que os três modos de semi-gradient SARSA convergem para a mesma solução.

---

### Lawn Mower (coverage path planning) — vários scripts

**Scripts:** `run_lawn_mower.py`, `run_lawn_mower_sarsa.py`, `run_lawn_mower_n_step_sarsa.py`, `run_lawn_mower_monte_carlo.py`

Problema de coverage path planning: o agente (corta-relva) deve visitar todas as células válidas do mapa `L_MAP` (mapa em L com obstáculos), minimizando revisitas e chegando ao estado terminal.

Cada script treina um agente diferente no mesmo ambiente:
- **`run_lawn_mower.py`** — Q-Learning (2000 episódios, epsilon decay de 1.0→0.05).
- **`run_lawn_mower_sarsa.py`** — SARSA on-policy.
- **`run_lawn_mower_n_step_sarsa.py`** — N-Step SARSA (n=4 por omissão).
- **`run_lawn_mower_monte_carlo.py`** — Monte Carlo Control (first-visit).

Todos geram em `outputs/lawn_mower/`:
- Curva de recompensa por episódio.
- Curva de comprimento de episódio.
- Heatmap de cobertura (frequência de visita por célula na política greedy final).

---

### TicTacToe (MCTS) — `experiments/tictactoe_mcts.py`

**Script:** `scripts/run_tictactoe_mcts.py`

Treina e avalia um `MCTSAgent` (1000 simulações por jogada) em TicTacToe:

1. Joga uma partida MCTS (X) vs aleatório (O) com render do tabuleiro.
2. Executa MCTS a partir de um estado a meio do jogo, visualizando a árvore de pesquisa (`plot_mcts_tree`) — visit counts, Q values e UCB scores por filho da raiz.
3. Avalia a taxa de vitória do MCTS vs 200 jogos aleatórios como X e como O.
4. Treina um `ReinforceAgent` e avalia MCTS vs REINFORCE, gerando gráfico comparativo em `outputs/tictactoe_mcts/`.

---

### TicTacToe (Policy Gradient) — `experiments/tictactoe_policygradient.py`

**Script:** `scripts/run_tictactoe_policygradient.py`

Demonstração completa de REINFORCE em TicTacToe:

1. Visualiza o encoding de features (27 dims) para um tabuleiro de exemplo, mostrando a representação perspetivo-relativa.
2. Treina `ReinforceAgent` por 100 000 episódios de self-play (com fração configurável de jogos contra adversário aleatório, `random_opp_frac=0.5`).
3. Avalia periodicamente (a cada 2000 episódios) a taxa de vitória como X e como O contra o adversário aleatório.
4. Gera curvas de aprendizagem (win rate, loss, entropia) em `outputs/tictactoe_policygradient/`.
5. Permite jogar contra o agente treinado via `play_game_vs_human`.

---

## 6. Comparação de Algoritmos por Use Case

---

### Blackjack — Predição de V(s)

Todos os agentes de predição operam com a mesma política fixa (threshold 20) e o mesmo número de episódios.

| | MC First-Visit | TD(0) | TD(n) (n=5) |
|---|---|---|---|
| **Atualização** | Fim de episódio (retorno completo) | Online por transição | Online por episódio (janela n) |
| **Bootstrapping** | Não | Sim (V(s')) | Sim (V(s_{t+n})) |
| **Variância** | Alta | Baixa | Intermédia |
| **Bias** | Nulo | Elevado (início) | Moderado |
| **Convergência em Blackjack** | Mais lenta (episódios curtos atenuam o problema) | Rápida em valor absoluto | Compromisso eficiente |
| **Uso de memória** | Episódio completo em RAM | Apenas transição atual | Janela de n transições |

Em Blackjack, os episódios são curtos (3–10 passos) e o sinal de recompensa só chega no fim (+1/-1), o que aproxima MC e TD(n) em comportamento. MC oferece estimativas sem bias mas com variância mais alta nos primeiros episódios; TD(0) converge mais rapidamente para uma estimativa mas pode ter bias inicial relevante. TD(5) é tipicamente o melhor compromisso neste ambiente.

---

### Windy GridWorld — Controlo Tabular

Três algoritmos on-policy aplicados ao mesmo ambiente de navegação:

| | Monte Carlo | SARSA | N-Step SARSA (n=4) |
|---|---|---|---|
| **Tipo** | Episódico | TD online | TD multi-step |
| **Atualização** | Fim de episódio | Por transição | Por janela de n passos |
| **Adequação a episódios longos** | Problemático (elevada variância com episódios de centenas de passos) | Excelente | Bom |
| **Velocidade de convergência** | Lenta (1000 episódios) | Rápida (500 episódios) | Moderada–rápida (5000 ep. para n=4) |
| **Propagação temporal** | Total (retorno completo) | Passo a passo | n passos |
| **Parâmetros** | ε | α, ε | α, ε, n |

MC é penalizado neste ambiente porque os episódios (antes de convergir) podem ter centenas de passos, gerando retornos de alta variância. SARSA converge mais rapidamente por atualizar online. N-Step SARSA com n=4 acelera a propagação de recompensas sem o custo de variância do MC, mas requer mais episódios para explorar suficientemente com ε baixo.

---

### Windy GridWorld — Controlo com Aproximação de Função

Comparação direta entre quatro variantes de SARSA linear no mesmo ambiente (script `run_windy_gridworld_torch_sarsa.py`):

| | Tabular SARSA | Linear SARSA (NumPy) | Torch SARSA (manual) | Torch SARSA (optimizer) |
|---|---|---|---|---|
| **Representação Q** | Tabela Q[(s,a)] | `w · φ(s,a)` (NumPy) | `w · φ(s,a)` (PyTorch) | `w · φ(s,a)` (PyTorch) |
| **Atualização** | Tabular in-place | Semi-gradiente manual | `loss.backward()` + update manual | `loss.backward()` + SGD |
| **Generalização** | Nenhuma | Por tile coding | Por tile coding | Por tile coding |
| **Escalabilidade** | Limitada a estados visitados | Generaliza para estados não vistos | Idem | Idem |
| **Resultado esperado** | Baselines | Convergência similar a tabular | Idêntico a NumPy | Idêntico a NumPy |

Os quatro agentes devem convergir para qualidades de política semelhantes no Windy GridWorld (ambiente discreto e de dimensão reduzida), já que a aproximação linear com tile coding representa exatamente o mesmo espaço de funções que a tabela. A diferença é pedagógica: demonstrar que PyTorch com `loss.backward()` e `SGD(lr=alpha)` executa matematicamente a mesma operação que a atualização NumPy manual.

---

### Lawn Mower — Coverage Path Planning

Quatro algoritmos de controlo aplicados ao mesmo mapa L:

| | Q-Learning | SARSA | N-Step SARSA | Monte Carlo |
|---|---|---|---|---|
| **Tipo** | Off-policy | On-policy | On-policy | On-policy (episódico) |
| **Adequação ao ambiente** | Boa (exploração independente da política greedy) | Boa | Boa | Limitada (episódios podem ser muito longos antes de cobrir o mapa) |
| **Epsilon decay** | Sim (1.0 → 0.05) | Fixo (0.1) | Fixo | Fixo |
| **Convergência esperada** | Boa após decay completo | Estável mas conservative | Rápida propagação | Lenta / requer muitos episódios |
| **Política aprendida** | Ótima (off-policy converge para π*) | Segura (evita penalidades de exploração) | Compromisso | Depende fortemente do número de episódios |

Q-Learning com epsilon decay é tipicamente o mais eficaz neste problema: a exploração agressiva inicial cobre todo o espaço, e o decay garante exploração da política ótima no final. SARSA aprende uma política ligeiramente mais conservadora (a exploração ε afeta a estimativa Q). Monte Carlo é o menos adequado para episódios longos como os de CPP.

---

### TicTacToe — Controlo Adversarial

Dois paradigmas fundamentalmente diferentes:

| | MCTS | REINFORCE |
|---|---|---|
| **Tipo** | Planeamento por simulação | Policy Gradient (aprendizagem) |
| **Pesos aprendidos** | Nenhum | θ ∈ R^{9×27} (política softmax linear) |
| **Requisitos por jogada** | 1000 simulações (ao vivo) | Forward pass em θ (instantâneo) |
| **Memória** | Árvore construída por jogo | Pesos θ (fixos após treino) |
| **Generalização** | Nenhuma (recomeça a cada jogo) | Total (via features perspetivo-relativas) |
| **Exploração** | Intrínseca via UCB1 | ε implícito via distribuição softmax |
| **Desempenho vs aleatório** | Próximo de 100% vitórias (com simulações suficientes) | Cresce com episódios de treino (típico: 70–85%) |
| **Custo computacional** | Elevado por jogada | Desprezível por jogada |
| **Treinável** | Não | Sim (self-play) |

MCTS é superiora com orçamento computacional suficiente: não erra jogadas óbvias e nunca perde com simulações adequadas. REINFORCE é mais eficiente em tempo de jogo e generaliza implicitamente pela estrutura de features — o mesmo θ joga X e O sem retreino, o que é conceptualmente elegante e computacionalmente eficiente.

O script `run_tictactoe_mcts.py` realiza a avaliação cruzada MCTS vs REINFORCE, permitindo observar empiricamente a diferença de desempenho entre um agente de planeamento perfeito e um agente de política aprendida.

---

*Rui Rodrigues — pg7942 — MIA RL 2026 — Universidade do Minho*
