# Reinforcement Learning no Windy GridWorld

# 1. Introdução

O objetivo desta experiência foi estudar diferentes algoritmos de controlo em ambientes estocásticos.

O Windy GridWorld introduz perturbações verticais ("vento") que alteram a posição do agente independentemente da ação executada.

Isto transforma o problema num ambiente:
- parcialmente imprevisível;
- sensível à política;
- adequado para estudar TD Learning.

---

# 2. Objetivos

O agente deveria:

- atingir o estado terminal;
- minimizar número de passos;
- aprender trajetórias robustas;
- adaptar-se ao vento.

---

# 3. Algoritmos Comparados

## SARSA

Método TD on-policy.

Atualização:

```text
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]
```

---

## N-Step SARSA

Extensão multi-step do SARSA.

Permite:
- propagação temporal mais rápida;
- melhor utilização do retorno futuro.

---

## Monte Carlo Control

Aprendizagem baseada em episódios completos.

Não utiliza bootstrapping.

---

## Linear SARSA

Aproximação linear:

```text
Q(s,a) = w · x(s,a)
```

---

## Torch SARSA

Utilização de redes neuronais para aproximação da função Q.

---

# 4. Reward Structure

## Movimento normal

```text
-1
```

## Objetivo alcançado

```text
+100
```

## Penalização temporal

Objetivo:
- incentivar trajetórias curtas.

---

# 5. Métricas Utilizadas

## Episode Length

Número médio de passos por episódio.

---

## Convergence Rate

Velocidade de estabilização da política.

---

## Average Reward

Reward médio acumulado.

---

# 6. Resultados Observados

## SARSA

### Comportamento
- aprendizagem estável;
- trajetórias conservadoras;
- adaptação robusta ao vento.

### Limitação
- convergência moderadamente lenta.

---

## N-Step SARSA

### Benefícios
- propagação mais rápida;
- aprendizagem acelerada;
- menor número médio de episódios.

### Limitação
- maior sensibilidade aos hiperparâmetros.

---

## Monte Carlo

### Benefícios
- elevada estabilidade final;
- ausência de bootstrapping.

### Limitações
- elevada variância;
- treino mais lento.

---

## Linear SARSA

### Benefícios
- generalização;
- melhor escalabilidade.

### Limitações
- dependência das features.

---

## Torch SARSA

### Benefícios
- elevada capacidade de generalização;
- melhor adaptação a estados complexos.

### Limitações
- maior custo computacional;
- treino menos estável inicialmente.

---

# 7. Comparação Experimental

| Algoritmo | Velocidade | Estabilidade | Escalabilidade | Variância |
|---|---|---|---|---|
| Monte Carlo | Baixa | Elevada | Média | Elevada |
| SARSA | Média | Muito boa | Média | Média |
| N-Step SARSA | Elevada | Boa | Média | Média |
| Linear SARSA | Elevada | Boa | Boa | Média |
| Torch SARSA | Muito elevada | Média | Muito elevada | Média |

---

# 8. Conclusão

Os resultados demonstraram:

- superioridade dos métodos TD em eficiência temporal;
- vantagens da aproximação de funções;
- robustez do SARSA em ambientes estocásticos;
- potencial do Deep RL para generalização.
