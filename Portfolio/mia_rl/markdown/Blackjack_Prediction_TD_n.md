# 🃏 Blackjack Prediction — TD(n) Implementation

## 📌 Objetivo

Foi estendida a implementação de métodos de *prediction* para o problema do Blackjack (agents\prediction\td.py), adicionando um novo agente baseado em **Temporal Difference de n passos (TD(n))**.

Já existiam dois métodos implementados:

* **First-Visit Monte Carlo (MC)**
* **TD(0)**

O trabalho consistiu em:

* Implementar o método TD(n)
* Integrá-lo na estrutura existente
* Comparar o seu comportamento com os outros métodos

---

## 🧠 Intuição dos Métodos

### 🔹 Monte Carlo (MC)

* Usa o retorno completo do episódio
* Não faz *bootstrapping*
* Alta variância
* Convergência mais lenta

### 🔹 TD(0)

* Usa apenas um passo no futuro
* Faz *bootstrapping* imediato
* Baixa variância
* Pode introduzir viés

### 🔹 TD(n)

* Usa **n passos reais + bootstrapping**
* Compromisso entre MC e TD(0)

---

## ⚙️ Implementação do TD(n)

Foi criada uma nova classe:

```python
class TDNPrediction(PredictionAgent)
```
```
agents\prediction\td.py
```

### Parâmetros principais:

* `n` → número de passos
* `alpha` → taxa de aprendizagem
* `gamma` → fator de desconto

---

## 🔁 Atualização dos Valores

Para cada estado no episódio, calculamos o retorno de n passos:

```math
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
```

### Passos da implementação:

1. Somar recompensas até n passos ou fim do episódio
2. Se ainda existir estado futuro → aplicar *bootstrapping*
3. Atualizar o valor do estado:

```python
V(s) ← V(s) + α (G - V(s))
```

---

## 🔌 Integração no Código

O novo agente foi integrado no script principal de experiências:
```
scripts\run_blackjack_prediction.py
```

### ✔️ Importação

```python
from mia_rl.agents.prediction import TDNPrediction
```

### ✔️ Criação do agente

```python
tdn_agent = TDNPrediction(n=5, alpha=args.td_alpha, gamma=1.0)
```

### ✔️ Treino

```python
tdn_history = train_prediction_agent(...)
```

---

## 📊 Comparação de Métodos

Foram gerados gráficos para:

* Função de valor (MC)
* Função de valor (TD(0))
* Função de valor (TD(n))
* Diferenças entre métodos

### Comparações feitas:

* TD(0) vs MC
* TD(n) vs MC
* TD(n) vs TD(0)

---

## 📈 Resultados Esperados

* **MC**: mais preciso a longo prazo, mas lento
* **TD(0)**: rápido, mas mais enviesado
* **TD(n)**: equilíbrio entre precisão e velocidade

---

## 🔍 Observações Importantes

* Quando `n = 1` → TD(n) ≈ TD(0)
* Quando `n` é grande → TD(n) aproxima Monte Carlo

Isto confirma a teoria de que TD(n) forma uma família de métodos intermédios.

---

## 📊 Análise dos Resultados

Com base nos gráficos obtidos, podemos comparar o comportamento dos três métodos:

### 🔹 Monte Carlo vs TD(0)

* Os resultados são bastante semelhantes (diferenças pequenas nos mapas)
* TD(0) aproxima bem o Monte Carlo
* No entanto, existem pequenas diferenças devido ao viés introduzido pelo bootstrap

### 🔹 TD(5) vs TD(0)

* As diferenças são muito reduzidas
* TD(5) apresenta resultados ligeiramente mais suaves

### 🔹 TD(5) vs Monte Carlo

* TD(5) aproxima-se mais do Monte Carlo do que TD(0)
* Ainda existem pequenas diferenças, mas o comportamento é intermédio

---

## ✅ Conclusão

* Todos os métodos aprendem corretamente a função de valor
* TD(0) aprende mais rápido, mas com algum viés
* Monte Carlo é mais preciso, mas com maior variância
* TD(n) (ex: n=5) oferece um compromisso entre os dois

👉 Quando aumentamos n, o comportamento aproxima-se do Monte Carlo
👉 Quando n=1, temos TD(0)

Assim, o TD(n) permite ajustar o equilíbrio entre estabilidade e precisão.

---

## 🚀 Possíveis Iterações

* Testar diferentes valores de `n`
* Ajustar `alpha`
* ...