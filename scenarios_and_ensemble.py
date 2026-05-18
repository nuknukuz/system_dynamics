"""
ensemble_lhs.py  —  LHS-ансамбль на отобранной тройке сценариев
Запуск:
    python ensemble_lhs.py
или в Colab:
    !python ensemble_lhs.py

Результаты:
    output/ensemble/fig_ensemble_corridors.png
    output/ensemble/{BAU|EVAV|Restrict}_<var>_quantiles.csv
    output/ensemble/robustness_table.csv
"""

import copy
import os
import time
from multiprocessing import Pool, cpu_count

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

from sd_model_v2 import simulate, P, initial_state

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

os.makedirs('output/ensemble', exist_ok=True)

# ──────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────────────────────

BASE_YEAR = 2000
END_YEAR = 2050
SIM_YEARS = END_YEAR - BASE_YEAR
N_OUTPUT = 251
YEARS = np.linspace(BASE_YEAR, END_YEAR, N_OUTPUT)
N_SAMPLES = 300

# Финальная тройка сценариев
SCENARIOS = {
    'BAU':      {},
    'EV+AV':    {'p_EV': 0.02, 'q_EV': 0.50, 'EV_local_clean': 0.95,
                 'p_AV': 0.015, 'AV_PT_bonus': 0.20},
    'Restrict': {'ParkCost': 25.0, 'delta_own': 0.4},
}

COLORS = {
    'BAU':      '#1f6e79',
    'EV+AV':    '#e07b39',
    'Restrict': '#7a5af8',
}

LABELS = {
    'BAU':      'BAU (базовый)',
    'EV+AV':    'EV+AV (технологический)',
    'Restrict': 'Restrict-strong (регуляторный)',
}

# Переменные для отслеживания
TRACK_VARS = [
    ('COWN', 's'),
    ('CONG', 'a'),
    ('CO2',  'a'),
    ('MSPT', 'a', 'MS_PT'),   # (key_in_model, src_dict, *alt_key)
    ('EV',   's'),
]

# Неопределённые параметры: (среднее, откл., нижняя, верхняя)
UNCERTAIN = {
    'V_sat_COWN':       (600,  60,   300,  900),
    'alpha_COWN':       (-1.81, 0.3, -3.0, -0.5),
    'beta_COWN':        (-0.08, 0.02, -0.20, -0.02),
    'IncomeShare_base': (0.65, 0.08, 0.40,  0.85),
    'alpha_GRP':        (0.06, 0.02,  0.02,  0.15),
    'Physical_cap_base':(18.0, 4.0,   8.0,  40.0),
    'mu':               (0.07, 0.02,  0.03,  0.15),
    'p_EV':             (0.005,0.003, 0.001, 0.03),
}

pnames_u = list(UNCERTAIN.keys())
k = len(pnames_u)

# ──────────────────────────────────────────────────────────────
# LHS-выборка
# ──────────────────────────────────────────────────────────────

sampler = qmc.LatinHypercube(d=k, seed=42)
lhs_unit = sampler.random(N_SAMPLES)
lhs_scaled = np.zeros_like(lhs_unit)

for i, pn in enumerate(pnames_u):
    lo, hi = UNCERTAIN[pn][2], UNCERTAIN[pn][3]
    lhs_scaled[:, i] = lo + lhs_unit[:, i] * (hi - lo)

print(f'LHS: {N_SAMPLES} точек × {k} параметров. '
      f'Неопределённые: {pnames_u}')


# ──────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────

def get_series(s_dct, a_dct, track_entry):
    """Извлечь массив по (key, src, *alt_key)."""
    key = track_entry[0]
    src = track_entry[1]
    alt_keys = list(track_entry[2:]) if len(track_entry) > 2 else []

    dct = s_dct if src == 's' else a_dct
    for k_try in [key] + alt_keys:
        if k_try in dct and dct[k_try] is not None:
            arr = np.asarray(dct[k_try], dtype=float)
            if arr.size > 0 and np.isfinite(arr).any():
                return arr
    return None


def run_sample(args):
    theta, sc_overrides = args
    p = copy.deepcopy(P)
    for pn, val in zip(pnames_u, theta):
        p[pn] = val
    p.update(sc_overrides)

    try:
        t, s, a = simulate(
            p,
            y0_override=initial_state(),
            years=SIM_YEARS,
            n_output=N_OUTPUT
        )
        result = {}
        for entry in TRACK_VARS:
            arr = get_series(s, a, entry)
            key = entry[0]
            if arr is not None and len(arr) == N_OUTPUT:
                result[key] = arr
            else:
                result[key] = np.full(N_OUTPUT, np.nan)
        return result
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# ПРОГОН
# ──────────────────────────────────────────────────────────────

all_results = {}

for sc_name, sc_overrides in SCENARIOS.items():
    print(f'\n  Сценарий {sc_name} ({N_SAMPLES} прогонов)...')
    t0 = time.time()

    args_list = [(lhs_scaled[i], sc_overrides) for i in range(N_SAMPLES)]

    with Pool(cpu_count()) as pool:
        runs = pool.map(run_sample, args_list)

    valid_runs = [r for r in runs if r is not None]
    nan_count = N_SAMPLES - len(valid_runs)
    elapsed = time.time() - t0

    print(f'    OK: {len(valid_runs)}/{N_SAMPLES}, ошибок: {nan_count}, '
          f'{elapsed:.1f}s')
    all_results[sc_name] = valid_runs


# ──────────────────────────────────────────────────────────────
# СОХРАНЕНИЕ CSV КВАНТИЛЕЙ
# ──────────────────────────────────────────────────────────────

print('\n  Сохранение CSV квантилей...')

for sc_name, runs in all_results.items():
    if not runs:
        continue
    for entry in TRACK_VARS:
        var = entry[0]
        arr = np.array([r[var] for r in runs if var in r and
                        np.isfinite(r[var]).any()], dtype=float)
        if arr.shape[0] < 10:
            continue
        n_t = min(arr.shape[1], N_OUTPUT)

        dfq = pd.DataFrame({
            'year': YEARS[:n_t],
            'q10':  np.nanpercentile(arr[:, :n_t], 10, axis=0),
            'q25':  np.nanpercentile(arr[:, :n_t], 25, axis=0),
            'q50':  np.nanpercentile(arr[:, :n_t], 50, axis=0),
            'q75':  np.nanpercentile(arr[:, :n_t], 75, axis=0),
            'q90':  np.nanpercentile(arr[:, :n_t], 90, axis=0),
        })

        sc_fname = sc_name.replace('+', '')
        fname = f'output/ensemble/{sc_fname}_{var}_quantiles.csv'
        dfq.to_csv(fname, index=False)


# ──────────────────────────────────────────────────────────────
# ТАБЛИЦА РОБАСТНОСТИ
# ──────────────────────────────────────────────────────────────

print('\n=== Таблица робастности (2050) ===')
header = f'{"Сценарий":10s}  {"Перем.":6s}  {"q50":>7s}  '
header += f'{"q10":>7s}  {"q90":>7s}  {"Ширина":>7s}  {"vs BAU (нет перекр.)":>20s}'
print(header)

# Сначала собираем квантили BAU
bau_q10, bau_q90 = {}, {}
bau_runs = all_results.get('BAU', [])
for entry in TRACK_VARS:
    var = entry[0]
    arr = np.array([r[var] for r in bau_runs if var in r], dtype=float)
    if arr.shape[0] >= 10:
        bau_q10[var] = float(np.nanpercentile(arr[:, -1], 10))
        bau_q90[var] = float(np.nanpercentile(arr[:, -1], 90))

rob_rows = []
for sc_name, runs in all_results.items():
    if not runs:
        continue
    for entry in TRACK_VARS:
        var = entry[0]
        arr = np.array([r[var] for r in runs if var in r], dtype=float)
        if arr.shape[0] < 10:
            continue

        q10_2050 = float(np.nanpercentile(arr[:, -1], 10))
        q50_2050 = float(np.nanpercentile(arr[:, -1], 50))
        q90_2050 = float(np.nanpercentile(arr[:, -1], 90))
        width = q90_2050 - q10_2050

        overlap = 'N/A'
        if sc_name != 'BAU' and var in bau_q10:
            bq10, bq90 = bau_q10[var], bau_q90[var]
            no_overlap = (q90_2050 < bq10) or (q10_2050 > bq90)
            overlap = '✓ нет' if no_overlap else '⚠ есть'

        print(f'{sc_name:10s}  {var:6s}  '
              f'{q50_2050:>7.4f}  {q10_2050:>7.4f}  {q90_2050:>7.4f}  '
              f'{width:>7.4f}  {overlap:>20s}')

        rob_rows.append({
            'Сценарий': sc_name,
            'Переменная': var,
            'q10_2050': round(q10_2050, 5),
            'q50_2050': round(q50_2050, 5),
            'q90_2050': round(q90_2050, 5),
            'Ширина_q10_q90': round(width, 5),
            'Перекрытие_с_BAU': overlap,
        })

df_rob = pd.DataFrame(rob_rows)
df_rob.to_csv('output/ensemble/robustness_table.csv', index=False)
print('\nSaved: output/ensemble/robustness_table.csv')


# ──────────────────────────────────────────────────────────────
# ГРАФИК КОРИДОРОВ
# ──────────────────────────────────────────────────────────────

plot_vars = [e[0] for e in TRACK_VARS]
n_rows = len(plot_vars)
n_cols = len(SCENARIOS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(5 * n_cols, 3.8 * n_rows),
    squeeze=False
)
fig.suptitle(
    f'LHS-ансамбль ({N_SAMPLES} прогонов): коридоры неопределённости 10–90%',
    fontsize=12
)

for j, (sc_name, sc_overrides) in enumerate(SCENARIOS.items()):
    runs = all_results.get(sc_name, [])

    for i, entry in enumerate(TRACK_VARS):
        var = entry[0]
        ax = axes[i][j]

        if not runs:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        arr = np.array([r[var] for r in runs if var in r], dtype=float)

        if arr.shape[0] < 10:
            ax.text(0.5, 0.5, f'Мало прогонов ({arr.shape[0]})',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        n_t = min(arr.shape[1], N_OUTPUT)
        yr = YEARS[:n_t]
        c = COLORS.get(sc_name, '#555')

        q10 = np.nanpercentile(arr[:, :n_t], 10, axis=0)
        q25 = np.nanpercentile(arr[:, :n_t], 25, axis=0)
        q50 = np.nanpercentile(arr[:, :n_t], 50, axis=0)
        q75 = np.nanpercentile(arr[:, :n_t], 75, axis=0)
        q90 = np.nanpercentile(arr[:, :n_t], 90, axis=0)

        ax.fill_between(yr, q10, q90, alpha=0.18, color=c)
        ax.fill_between(yr, q25, q75, alpha=0.38, color=c)
        ax.plot(yr, q50, color=c, lw=2.5, label='медиана (q50)')
        ax.plot(yr, q10, color=c, lw=0.8, ls='--', alpha=0.6)
        ax.plot(yr, q90, color=c, lw=0.8, ls='--', alpha=0.6, label='q10/q90')

        # Ширина коридора в 2050
        width_2050 = float(q90[-1] - q10[-1])
        ax.text(
            0.97, 0.05,
            f'Δ₂₀₅₀ = {width_2050:.4f}',
            transform=ax.transAxes, ha='right', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', lw=0.6)
        )

        if i == 0:
            ax.set_title(LABELS.get(sc_name, sc_name), fontsize=9.5)
        if j == 0:
            ax.set_ylabel(var, fontsize=9)
        ax.set_xlabel('Год', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='upper right')

plt.tight_layout(pad=1.2)
plt.savefig('output/ensemble/fig_ensemble_corridors.png',
            dpi=200, bbox_inches='tight')
plt.close()

print('Saved: output/ensemble/fig_ensemble_corridors.png')
print('\n=== ГОТОВО ===')
print('Файлы:')
print('  output/ensemble/fig_ensemble_corridors.png')
print('  output/ensemble/robustness_table.csv')
print('  output/ensemble/{BAU|EVAV|Restrict}_*_quantiles.csv')
