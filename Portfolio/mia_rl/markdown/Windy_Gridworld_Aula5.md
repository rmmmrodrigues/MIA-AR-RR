# Explicação da Prática 5 — SARSA e Windy Gridworld

## Objetivo da ficha

Nesta ficha foi implementado controlo **on-policy** usando o algoritmo:

* SARSA
* n-step SARSA (exercício de portefólio)

O ambiente utilizado foi o:

* Windy Gridworld

O objetivo do agente é aprender uma política que consiga chegar ao objetivo no menor número possível de passos, apesar do vento existente em determinadas colunas.

---

# 1. Windy Gridworld

Ficheiro:
`windy_gridworld.py`

---

## O que é o Windy Gridworld

O Windy Gridworld é um ambiente clássico de Reinforcement Learning apresentado no livro de Sutton & Barto.

O ambiente consiste numa grelha onde:

* o agente se move em 4 direções
* algumas colunas possuem vento vertical
* o vento empurra o agente para cima
* cada passo tem recompensa negativa
* o objetivo é alcançar a célula final

---

# Estrutura do ambiente

## Estados

Cada estado é representado por:

```python
(row, col)
```

Exemplo:

```python
(3, 4)
```

---

## Ações

As ações disponíveis são:

```python
("up", "down", "left", "right")
```

Cada ação possui um deslocamento:

```python
ACTION_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
```

---

# Método `step_from_state()`

## Parte principal da prática

Este método implementa toda a dinâmica do ambiente.

---

## 1. Validar ação

```python
if action not in ACTIONS:
    raise ValueError(...)
```

Garante que apenas ações válidas são executadas.

---

## 2. Obter posição atual

```python
row, col = state
```

---

## 3. Aplicar movimento da ação

```python
delta_row, delta_col = ACTION_TO_DELTA[action]
```

Exemplo:

* `up` → `(-1,0)`
* `right` → `(0,1)`

---

## 4. Aplicar vento

```python
wind_strength = self.wind[col]
```

O vento depende da coluna atual.

Quanto maior o valor:

* mais o agente sobe automaticamente.

---

## 5. Calcular próximo estado

```python
next_row = min(max(row + delta_row - wind_strength, 0), self.rows - 1)
next_col = min(max(col + delta_col, 0), self.cols - 1)
```

Aqui:

* aplica-se o movimento
* aplica-se o vento
* limita-se o agente aos limites da grelha

---

## 6. Estado terminal

```python
done = next_state == self.goal
```

O episódio termina quando o agente chega ao objetivo.

---

## 7. Recompensa

```python
return next_state, self.reward_per_step, done
```

Cada passo tem normalmente:

```python
-1
```

Isto incentiva o agente a encontrar caminhos mais curtos.

---

# 2. SARSA

Ficheiro:
`sarsa.py`

---

# O que é SARSA

SARSA é um algoritmo:

* temporal-difference
* on-policy
* de controlo

O nome vem de:

[
(S, A, R, S', A')
]

Porque o update utiliza:

* estado atual
* ação atual
* recompensa
* próximo estado
* próxima ação

---

# Fórmula do SARSA

[
Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]
]

Onde:

* (\alpha) → learning rate
* (\gamma) → discount factor
* (Q(s,a)) → valor atual
* (Q(s',a')) → bootstrap

---

# Política ε-greedy

## Método `select_action()`

```python
if self.rng.random() < self.epsilon:
```

Com probabilidade ε:

* escolhe ação aleatória

Caso contrário:

* escolhe a melhor ação conhecida.

---

# Update SARSA

## 1. Bootstrap

```python
bootstrap = self.action_value_of(transition.next_state, next_action)
```

Obtém:

[
Q(s',a')
]

---

## 2. Calcular target

```python
td_target = transition.reward + self.gamma * bootstrap
```

Equivalente a:

[
r + \gamma Q(s',a')
]

---

## 3. Atualizar Q-value

```python
self.Q[(state, action)] = current + alpha * (target - current)
```

Implementa:

[
Q(s,a) \leftarrow Q(s,a) + \alpha(TDerror)
]

---

# 3. Exercício de Portefólio — n-step SARSA

Ficheiro:
`n_step_sarsa.py`

---

# Objetivo do exercício

Generalizar o SARSA clássico para utilizar:

* múltiplas recompensas futuras
* atualização após n passos

Em vez de atualizar usando apenas:

[
r + \gamma Q(s',a')
]

utiliza-se:

[
G_t^{(n)}
]

ou seja:

* retorno acumulado de n passos.

---

# Conceito principal

No SARSA normal:

```text
1-step return
```

No n-step SARSA:

```text
n-step return
```

O agente espera mais tempo antes de atualizar.

---

# Estrutura do algoritmo

## Buffer de transições

```python
self._transitions
```

Guarda:

* estados
* ações
* recompensas

até existir informação suficiente para calcular o retorno n-step.

---

# Método `update_transition()`

## 1. Guardar transição

```python
self._transitions.append(transition)
```

---

## 2. Verificar se existem n passos

```python
if len(self._transitions) >= self.n_steps:
```

Quando existem transições suficientes:

* atualiza o estado mais antigo.

---

## 3. Episódio terminal

```python
if transition.done:
    self.end_episode()
```

No final do episódio:

* faz flush das transições restantes.

---

# Método `_update_oldest()`

## Parte mais importante do exercício

Este método calcula o retorno n-step.

---

## 1. Horizonte efetivo

```python
horizon = min(self.n_steps, len(self._transitions))
```

No fim do episódio pode haver menos de `n` transições.

---

## 2. Calcular retorno acumulado

```python
for idx in range(horizon):
    target += (self.gamma**idx) * self._transitions[idx].reward
```

Implementa:

[
R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...
]

---

## 3. Bootstrap opcional

```python
if use_bootstrap:
```

Se ainda não terminou:

```python
Q(s_{t+n}, a_{t+n})
```

é usado.

---

## 4. Atualização do Q-value

```python
self.Q[(state, action)] = current + alpha * (target - current)
```

Atualiza o estado mais antigo do buffer.

---

# Porque o n-step SARSA é importante

O n-step SARSA cria um equilíbrio entre:

* Monte Carlo
* Temporal Difference

---

## SARSA clássico

* menor variância
* mais bias

---

## Monte Carlo

* menor bias
* maior variância

---

## n-step SARSA

Fica entre os dois extremos.

---

# Scripts de execução

## SARSA normal

```bash
python -m mia_rl.scripts.run_windy_gridworld_sarsa
```

---

## n-step SARSA

```bash
python -m mia_rl.scripts.run_windy_gridworld_n_step_sarsa
```

---

# Resultados esperados

Após treino:

* o agente aprende um caminho eficiente
* evita movimentos desnecessários
* compensa o efeito do vento

Os gráficos mostram:

* diminuição do número de passos
* melhoria da política
* convergência do treino

---

# Conceitos importantes aprendidos

## 1. Reinforcement Learning On-Policy

O agente aprende usando a mesma política usada para agir.

---

## 2. Temporal Difference Learning

Aprendizagem baseada em:

* recompensas imediatas
* bootstrap

---

## 3. Exploração vs Exploitation

ε-greedy permite:

* explorar ações novas
* explorar conhecimento existente

---

## 4. Multi-step Returns

n-step SARSA utiliza várias recompensas futuras.

---

# Conclusão

Esta ficha introduz os primeiros algoritmos de controlo temporal-difference.

O SARSA permite:

* aprender políticas diretamente
* melhorar comportamento online
* resolver problemas sequenciais

O exercício de portefólio (n-step SARSA) é especialmente importante porque introduz:

* multi-step bootstrapping
* retornos acumulados
* ligação entre TD e Monte Carlo

Estes conceitos são fundamentais para algoritmos mais avançados de Reinforcement Learning.
