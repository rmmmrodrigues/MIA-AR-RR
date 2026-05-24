# Reinforcement Learning para Coverage Path Planning de um Corta-Relva Autónomo

# 1. Introdução

Este projeto teve como objetivo desenvolver um ambiente de Reinforcement Learning (RL) para um problema de coverage path planning (CPP), inspirado no comportamento de um corta-relva autónomo.

O problema consiste em:

* percorrer todas as células válidas do ambiente;
* minimizar revisitas;
* evitar obstáculos;
* atingir o estado final;
* aprender uma política eficiente.

O projeto foi desenvolvido reutilizando a arquitetura existente do Gridworld.

---

# 2. Estrutura Inicial do Projeto

A estrutura original do projeto era:

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
    run_gridworld.py
```

O Gridworld já incluía:

* ambientes tabulares;
* value iteration;
* policy iteration;
* plotting;
* políticas;
* Bellman backups.

---

# 3. Estratégia de Desenvolvimento

Inicialmente foi considerada a possibilidade de modificar diretamente o Gridworld original.

No entanto, essa abordagem poderia:

* criar problemas em implementações anteriores;
* introduzir incompatibilidades;
* aumentar complexidade;
* dificultar manutenção.

Por esse motivo foi decidido:

Manter o Gridworld original intacto

E criar um novo ambiente:

```text
envs/lawn_mower.py
```

Desta forma:

* o Gridworld continuou funcional;
* o novo problema ficou modular;
* a arquitetura manteve-se limpa.

---

# 4. Estrutura Final do Projeto

```text
mia_rl/

  agents/
    control/
      q_learning.py

    planning/
      gridworld.py

  envs/
    gridworld.py
    lawn_mower.py

  experiments/
    lawn_mower.py

  plots/
    lawn_mower.py

  scripts/
    run_lawn_mower.py

  outputs/
    lawn_mower/
```

---

# 5. Reutilização do Gridworld

Foram reutilizados alguns componentes do Gridworld original:

## Ações

```python
ACTIONS = ["U", "D", "L", "R"]
```

## Movimentos

```python
ACTION_TO_DELTA = {
    "U": (-1, 0),
    "D": ( 1, 0),
    "L": ( 0,-1),
    "R": ( 0, 1),
}
```

## Estrutura do método `step()`

## Estrutura do ambiente tabular

## Estrutura de plotting

## Organização do projeto

---

# 6. Problema do Coverage Path Planning

Ao contrário do Gridworld clássico, onde o objetivo é apenas chegar ao terminal, neste problema o agente precisa:

* visitar todas as células válidas;
* minimizar revisitas;
* otimizar a trajetória.

Isso transforma o problema num:

# Coverage Optimization Problem

---

# 7. Mapa em L

Foi criado um ambiente irregular em formato de L.

## Legenda do mapa

```text
0 = obstáculo / inválido
1 = célula livre
2 = início
3 = objetivo final
```

## Exemplo do mapa

```python
L_MAP = np.array([
    [2,1,1,1,1,1,1,1,1],
    [1,0,1,0,1,0,1,0,1],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,1],
    [1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,1,1,3],
])
```

---

# 8. Representação do Estado

## Gridworld clássico

O estado era apenas:

```python
state = (r,c)
```

---

## Lawn Mower

O agente precisa de memória de cobertura.

O estado passou a ser:

```python
state = (
    position,
    visited
)
```

Exemplo:

```python
(
    (2,3),
    frozenset([
        (0,0),
        (0,1),
        (1,1),
    ])
)
```

---

# 9. Memória de Cobertura

Foi utilizado:

```python
frozenset
```

para guardar células visitadas.

Isto permite:

* hashing;
* utilização como chave da Q-table;
* compatibilidade com dicionários.

---

# 10. Reward Shaping

O reward shaping foi essencial para guiar o comportamento do agente.

## Recompensas utilizadas

### Nova célula

```python
+5.0
```

### Revisita

```python
-2.0
```

### Movimento inválido

```python
-5.0
```

### Penalização por movimento

```python
-0.1
```

### Cobertura completa

```python
+100.0
```

---

# 11. Escolha do Algoritmo

Inicialmente foi considerada:

* Value Iteration;
* Policy Iteration.

No entanto, o espaço de estados cresceu exponencialmente devido à memória de cobertura.

O número de estados tornou-se:

```text
posição × subconjuntos visitados
```

Isso inviabilizou Dynamic Programming clássico.

---

# 12. Q-Learning

Foi então implementado:

# Q-Learning tabular

## Vantagens

* aprendizagem online;
* não necessita enumerar estados;
* adequado para coverage tasks;
* simples de implementar.

---

# 13. Estrutura da Q-Table

A Q-table foi implementada com:

```python
defaultdict(lambda: defaultdict(float))
```

A chave do estado ficou:

```python
(position, visited)
```

---

# 14. Política Epsilon-Greedy

Foi utilizada uma política epsilon-greedy.

## Exploração

```python
random.random() < epsilon
```

## Exploração decrescente

```python
epsilon_decay = 0.995
```

## Valor mínimo

```python
epsilon_min = 0.05
```

---

# 15. Treino

O treino foi realizado por episódios.

## Configuração

```python
episodes = 2000
max_steps = 500
```

---

# 16. Métricas Utilizadas

## Coverage Ratio

```python
coverage_ratio = len(visited) / env.n_valid_cells
```

Representa:

* percentagem de cobertura do ambiente.

---

## Repeated Visits

```python
repeated_visits = (
    sum(visit_counts.values())
    -
    len(visit_counts)
)
```

Representa:

* número de revisitas.

---

## Efficiency

```python
efficiency = unique_cells / total_steps
```

Representa:

* eficiência da cobertura.

Valores próximos de:

```text
1.0
```

indicam cobertura muito eficiente.

---

# 17. Heatmap de Coverage

Foi implementado um heatmap para visualizar:

* intensidade de revisitas;
* zonas problemáticas;
* hotspots;
* qualidade espacial da política.

## Interpretação das cores

### Verde

Poucas visitas.

Cobertura eficiente.

---

### Amarelo

Algumas revisitas.

Normal em corredores.

---

### Vermelho

Muitas revisitas.

Pode indicar:

* loops;
* exploração ineficiente;
* reward shaping inadequado.

---

# 18. Trajetória Final

Foi adicionada uma visualização da trajetória completa.

A trajetória permitiu:

* analisar o comportamento espacial;
* identificar loops;
* observar padrões de exploração;
* validar a política aprendida.

---

# 19. Visualização Final

O plot final passou a incluir:

* heatmap;
* trajetória;
* posição inicial;
* posição final;
* contagem de visitas;
* obstáculos.

---

# 20. Persistência de Resultados

As figuras passaram a ser guardadas automaticamente.

## Função utilizada

```python
def save_plot(fig, output_dir: Path, filename: str) -> None:
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
```

---

# 21. Outputs Gerados

```text
outputs/
    lawn_mower/

        training_rewards.png
        coverage_heatmap.png
```

---

# 22. Principais Desafios

## Explosão do espaço de estados

A introdução da memória de cobertura aumentou a complexidade.

---

## Revisitas excessivas

Foi necessário ajustar reward shaping.

---

## Balanceamento exploração/explotação

O ajuste do epsilon decay foi importante para estabilizar aprendizagem.

---

# 23. Resultados Obtidos

O agente conseguiu:

* aprender políticas de coverage;
* minimizar revisitas;
* evitar obstáculos;
* completar cobertura do ambiente.

O heatmap demonstrou:

* melhoria progressiva da política;
* redução de hotspots;
* cobertura espacial mais eficiente.

---

# 24. Melhorias Futuras

## Otimização do estado

Substituir:

```python
frozenset
```

por:

```python
bitmask
```

para melhorar:

* memória;
* velocidade;
* escalabilidade.

---

## Deep Reinforcement Learning

Possíveis evoluções:

* DQN;
* Double DQN;
* PPO;
* Actor-Critic.

---

## Multi-Agent Coverage

Adicionar:

* múltiplos corta-relvas;
* coordenação;
* divisão do espaço.

---

## Ambientes Dinâmicos

Adicionar:

* obstáculos móveis;
* terreno estocástico;
* bateria limitada.

---

# 25. Conclusão

O projeto permitiu transformar um Gridworld clássico num problema realista de Coverage Path Planning utilizando Reinforcement Learning.

A reutilização da arquitetura original permitiu:

* manter modularidade;
* preservar implementações anteriores;
* acelerar desenvolvimento.

O sistema final incluiu:

* ambiente irregular;
* memória de cobertura;
* reward shaping;
* Q-learning;
* heatmaps;
* análise espacial;
* métricas de eficiência.

O projeto pode ser aproveitado para:

* aspiradores robots;
* agricultura autónoma;
* ...;
