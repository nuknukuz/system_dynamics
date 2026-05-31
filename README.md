# system_dynamics

Основные блоки кода, используемые при написании ВКР «Исследование методов моделирования
динамики сложных систем на примере транспортной системы города».

## Структура

| Файл | Назначение |
|------|-----------|
| `sd_model_v2.py` | Ядро SD-модели ВАТС: 14 стоков, ~62 переменных, правые части (`f`), вспомогательные величины (`compute_aux`), функция `simulate`. |
| `calibration.py` | Калибровка поведенческих параметров под город методом Differential Evolution. Запуск: `python calibration.py --city shenzhen` / `--city singapore`. |
| `cross_city_test.py` | Тест переносимости (cross-city transfer test): калибровка на доноре, перенос параметров на рецепиента с его начальными условиями, NRMSE. См. ниже. |
| `validation_tests.py` | Структурная валидация: extreme conditions + behaviour reproduction (induced demand, Downs–Thomson, S-кривая EV). |
| `scenarios.py`, `scenarios2.py`, `scenarios_and_ensemble.py` | Сценарные прогоны и ансамблевый (LHS) анализ. |
| `calibration_data/*.csv` | Реальные временны́е ряды городов (Шэньчжэнь, Сингапур), 2000–2020. Источник — Google Sheets `*_timeseries`. |
| `_build_calib_data.py` | Воспроизводимо формирует `calibration_data/*.csv` из исходных данных. |

## Тест переносимости (cross_city_test.py)

Реализует **transfer scaling** (классификация TRB 2014): поведенческие параметры
фиксируются по городу-донору, а начальные условия берутся из реальных данных
города-рецепиента. Это НЕ «наивный перенос» — начальные условия не копируются у донора.

```bash
python cross_city_test.py                              # Шэньчжэнь -> Сингапур, быстрый режим
python cross_city_test.py --maxiter 200 --popsize 15   # полная калибровка (как в дипломе)
python cross_city_test.py --donor singapore --recipient shenzhen
```

Высокий NRMSE переноса (> ~25%) — ожидаемый и информативный результат: унифицированная
структура переносима по форме, но значения требуют повторной подгонки (ср. MARS,
который рекалибруется в каждом из ~20 городов).

## Интегратор

`simulate(..., method=...)` поддерживает:
- `"RK4"` — явный Рунге–Кутта 4-го порядка с фиксированным шагом (`dt=0.25` по умолч.), как заявлено в тексте ВКР;
- `"LSODA"` / `"RK45"` / др. — через `scipy.integrate.solve_ivp` (`max_step=0.5`).

Результаты RK4 (Δt=0.25) и LSODA совпадают с точностью ~0.002% по всем стокам — модель численно устойчива, выбор интегратора на выводы не влияет.

```python
from sd_model_v2 import simulate, P
t, stocks, aux = simulate(P, years=20, n_output=81, method="RK4", dt=0.25)
```
