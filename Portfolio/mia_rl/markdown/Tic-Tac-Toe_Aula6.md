# Explicação da Prática 6 — Aprendizagem por Reforço

## Objetivo da ficha

Esta ficha introduz **aproximação de função linear** em Reinforcement Learning usando:

* **SARSA linear (NumPy)**
* **SARSA linear (PyTorch)**
* **Tile Coding**
* Implementação de um ambiente simples de jogo: **Tic-Tac-Toe**

A principal diferença relativamente às práticas anteriores é que deixamos de usar tabelas (`Q-table`) e passamos a aproximar a função de valor através de:

[
\hat{q}(s,a) = w \cdot \phi(s,a)
]

Onde:

* (w) → vetor de pesos
* (\phi(s,a)) → vetor de features do estado-ação

---

# 1. Linear SARSA (NumPy)

Ficheiro:
`linear_sarsa.py`

## Objetivo

Implementar o algoritmo **Semi-Gradient SARSA** usando apenas NumPy.

A atualização usada é:

[
w \leftarrow w + \alpha \delta \phi(s,a)
]

onde:

[
\delta = r + \gamma \hat{q}(s',a') - \hat{q}(s,a)
]

---

## Explicação do método `update_transition`

### 1. Obter as features

```python
phi = self.phi(transition.state, transition.action)
```

A função `phi` transforma o par `(estado, ação)` num vetor numérico.

---

### 2. Bootstrap

```python
if not transition.done and transition.next_state is not None:
    next_action = self._selected_actions[transition.next_state]
    bootstrap = self.action_value_of(transition.next_state, next_action)
else:
    bootstrap = 0.0
```

Se o episódio não terminou:

[
\hat{q}(s',a')
]

é usado para calcular o target.

Caso contrário:

[
bootstrap = 0
]

---

### 3. Cálculo do erro TD

```python
delta = (
    transition.reward
    + self.gamma * bootstrap
    - float(self.w @ phi)
)
```

Aqui calculamos:

[
\delta = r + \gamma q(s',a') - q(s,a)
]

---

### 4. Atualização dos pesos

```python
self.w += self.alpha * delta * phi
```

Cada peso é ajustado proporcionalmente ao erro TD.

---

### 5. Guardar erro TD

```python
self._td_errors.append(abs(delta))
```

Isto permite analisar a evolução do treino.

---

# 2. Torch SARSA (PyTorch)

Ficheiro:
`torch_sarsa.py`

---

## Objetivo

Implementar exatamente o mesmo algoritmo mas usando:

* autograd
* tensors
* SGD optimizer

O modelo usado é:

```python
nn.Linear(n_features, 1, bias=False)
```

Ou seja:

[
\hat{q}(s,a) = w \cdot \phi(s,a)
]

---

## Fluxo do treino

### 1. Converter features para tensor

```python
phi = self._to_tensor(transition.state, transition.action)
```

---

### 2. Limpar gradientes

```python
self.optimizer.zero_grad()
```

Em PyTorch os gradientes acumulam automaticamente.

---

### 3. Forward pass

```python
pred = self.model(phi)
```

Calcula:

[
\hat{q}(s,a)
]

---

### 4. Construção do target

```python
target_tensor = torch.tensor([target], dtype=torch.float32)
```

O target é desacoplado do grafo computacional.

Isto implementa o conceito de:

* **semi-gradient**
* sem propagação de gradientes para o bootstrap

---

### 5. Loss

```python
loss = 0.5 * F.mse_loss(pred, target_tensor)
```

Equivalente a:

[
\frac{1}{2}(pred-target)^2
]

O fator `0.5` simplifica a derivada.

---

### 6. Backpropagation

```python
loss.backward()
```

PyTorch calcula automaticamente:

[
\nabla_w
]

---

### 7. Atualização SGD

```python
self.optimizer.step()
```

Atualiza:

[
w \leftarrow w - \alpha \nabla_w
]

---

# 3. Exercício do Portefólio — Tic-Tac-Toe

Ficheiro:
`tictactoe.py`

---

# Objetivo do exercício

Construir um ambiente RL completo para o jogo Tic-Tac-Toe.

O ambiente segue a interface:

* `reset`
* `step`
* `available_actions`
* `is_terminal`
* `render`

---

# Representação do tabuleiro

O estado é:

```python
tuple[int, ...]
```

com 9 posições:

```text
0 1 2
3 4 5
6 7 8
```

Valores:

* `0` → vazio
* `1` → jogador X
* `-1` → jogador O

Exemplo:

```python
(1, 0, -1,
 0, 1, 0,
 -1, 0, 0)
```

---

# Método `reset()`

## Objetivo

Reiniciar o jogo.

```python
self.board = (0,) * 9
self.current_player = 1
```

* limpa o tabuleiro
* define X como primeiro jogador

---

# Método `available_actions(state)`

## Objetivo

Devolver todas as casas vazias.

```python
return [i for i, cell in enumerate(state) if cell == 0]
```

Exemplo:

```python
(1,0,-1,
 0,1,0,
 -1,0,0)
```

Ações disponíveis:

```python
[1,3,5,7,8]
```

---

# Método `is_terminal(state)`

## Objetivo

Verificar se o jogo terminou.

O jogo termina quando:

* existe vencedor
* não existem casas vazias

```python
return _winner(state) != 0 or 0 not in state
```

---

# Método `step(action)`

## Parte mais importante do exercício

Este método implementa toda a dinâmica do jogo.

---

## 1. Validar jogada

```python
if self.board[action] != 0:
    raise ValueError(...)
```

Impede jogar numa célula ocupada.

---

## 2. Atualizar tabuleiro

```python
board[action] = self.current_player
```

Coloca:

* `1` para X
* `-1` para O

---

## 3. Verificar vencedor

```python
winner = _winner(new_board)
```

A função `_winner` verifica:

* linhas
* colunas
* diagonais

---

## 4. Verificar terminal

```python
done = winner != 0 or 0 not in new_board
```

---

## 5. Calcular recompensa

```python
reward = 1.0 if winner == self.current_player else 0.0
```

Recompensa:

* `+1` → vitória
* `0` → empate ou jogo continua

---

## 6. Trocar jogador

```python
self.current_player *= -1
```

Alterna:

* `1 → -1`
* `-1 → 1`

---

## 7. Atualizar estado interno

```python
self.board = new_board
```

---

## 8. Retornar transição

```python
return new_board, reward, done
```

Formato padrão de ambientes RL.

---

# Método `render()`

## Objetivo

Mostrar o tabuleiro no terminal.

Exemplo:

```text
 X | 2 | O
---+---+---
 4 | X | 6
---+---+---
 O | 8 | 9
```

Casas vazias mostram o número da ação disponível.

---

# Conceitos importantes aprendidos

## 1. Aproximação de Função

Em vez de guardar valores numa tabela:

[
Q(s,a)
]

aproximamos:

[
\hat{Q}(s,a)
]

---

## 2. Features

Estados são convertidos em vetores numéricos:

[
\phi(s,a)
]

---

## 3. Semi-Gradient Methods

O bootstrap:

[
q(s',a')
]

não recebe gradientes.

---

## 4. Reinforcement Learning em jogos

O Tic-Tac-Toe mostra:

* estados
* ações
* recompensas
* transições
* terminação

num ambiente simples.

---

# Conclusão

Esta ficha introduz uma das transições mais importantes em RL:

* de métodos tabulares
* para aproximação de função

Além disso, o exercício do Tic-Tac-Toe ajuda a compreender como construir um ambiente RL completo desde raiz, incluindo:

* dinâmica do jogo
* gestão de estados
* recompensas
* terminalidade
* renderização

É uma base importante para ambientes mais complexos usados em Deep Reinforcement Learning.
