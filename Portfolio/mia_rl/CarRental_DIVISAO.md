# Divisao do MDP CarRental

Este ficheiro documenta a divisao do notebook `notebooks/CarRental.ipynb` pela arquitetura do pacote `mia_rl`, seguindo o mesmo principio usado em `MDP_GridWorld.ipynb`: o notebook passa a importar codigo do pacote em vez de concentrar todas as classes e funcoes nas celulas.

## Estrutura criada

```text
mia_rl/
  envs/
    carrental.py

  agents/
    planning/
      carrental.py

  experiments/
    carrental.py

  plots/
    carrental.py

  scripts/
    run_carrental.py
    run_carrental.bat
```

---

## Divisao dos blocos

### `mia_rl/envs/carrental.py`

Contém a definição do problema (modelo MDP):

* `CarRentalParams` — parâmetros do problema
* `CarRentalMDP` — dinâmica do ambiente
* `poisson_pmf_truncated` — distribuição de probabilidade
* Métodos do modelo:

  * `states`
  * `possible_actions`
  * `after_move`
  * `expected_transition`
  * `_loc_outcomes` (cache de dinâmica por localização)

👉 Este módulo define apenas o **mundo (dinâmica e probabilidades)**.

---

### `mia_rl/agents/planning/carrental.py`

Contém os algoritmos de programação dinâmica (DP):

* Funções auxiliares:

  * `q_from_v`
  * `bellman_expectation_backup_v`
  * `bellman_optimality_backup_v`
  * `zeros_V`

* Algoritmos:

  * `policy_evaluation`
  * `policy_improvement`
  * `policy_iteration`
  * `value_iteration`

👉 Este módulo implementa o **solver (planeamento com modelo conhecido)**.

---

### `mia_rl/experiments/carrental.py`

Contém a lógica de execução dos experimentos:

* `run_carrental(output_dir, gamma=0.9)`

  * cria o ambiente (`CarRentalMDP`)
  * executa:

    * policy iteration
    * value iteration
  * gera gráficos usando o módulo `plots`
  * guarda resultados em `outputs/carrental`

👉 Este módulo separa a **lógica experimental** da execução.

---

### `mia_rl/plots/carrental.py`

Contém funções de visualização:

* `policy_to_array`
* `plot_policy`
* `plot_values`

👉 Responsável apenas por **visualização dos resultados**.

---

### `mia_rl/scripts/run_carrental.py`

Script executável:

* configura o `PYTHONPATH`
* define `output_dir`
* chama `run_carrental`

👉 Funciona como **entry point leve**, delegando toda a lógica para `experiments`.

---

### `mia_rl/scripts/run_carrental.bat`

Script auxiliar para execução em Windows:

* chama `run_carrental.py`
* permite execução rápida via PowerShell

---

## Como executar

Em PowerShell:

```powershell
python mia_rl/scripts/run_carrental.py
```

Ou via script:

```powershell
& .\mia_rl\scripts\run_carrental.bat
```

---

## Notas

* Separação clara entre responsabilidades:

  * **envs** → definição do problema
  * **planning** → algoritmos de resolução (DP)
  * **experiments** → execução e organização de resultados
  * **plots** → visualização
  * **scripts** → ponto de entrada

* Mantém consistência com a organização usada no Gridworld.

* Evita dependências cruzadas (ex: plots dentro de envs ou DP).

* Estrutura escalável para novos ambientes e algoritmos.
