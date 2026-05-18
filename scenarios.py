import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sd_model_v2 import simulate, P, initial_state

os.makedirs('output/scenarios', exist_ok=True)

# 1) Правильные имена параметров
SCENARIOS = {
    'BAU': {},
    'EV+AV': {
        'p_EV': 0.02,
        'q_EV': 0.50,
        'EV_local_clean': 0.95,
        'p_AV': 0.015,
        'AV_PT_bonus': 0.20,
    },
    'Restrict': {
        'ParkCost': 8.0,
        'delta_own': 0.40,
    },
}

# 2) simulate() получает ЧИСЛО лет, а не массив
BASE_YEAR = 2000
END_YEAR = 2050
SIM_YEARS = END_YEAR - BASE_YEAR
N_OUTPUT = 251
YEARS = np.linspace(BASE_YEAR, END_YEAR, N_OUTPUT)

COLORS = {
    'BAU': '#1f6e79',
    'EV+AV': '#e07b39',
    'Restrict': '#7a5af8',
}

def get_series(dct, *keys):
    for k in keys:
        if k in dct and dct[k] is not None:
            arr = np.asarray(dct[k], dtype=float)
            if len(arr) > 0:
                return arr
    return None

def run_scenario(name, overrides):
    p = copy.deepcopy(P)
    p.update(overrides)

    print(f'\n=== {name} ===')
    for k, v in overrides.items():
        print(f'{k:20s} -> {"OK" if k in P else "MISSING"} | {v}')

    t, s, a = simulate(
        p,
        y0_override=initial_state(),
        years=SIM_YEARS,
        n_output=N_OUTPUT
    )

    print('state keys sample:', list(s.keys())[:15])
    print('aux keys sample:  ', list(a.keys())[:20])

    cown = get_series(s, 'COWN')
    cong = get_series(a, 'CONG')
    co2  = get_series(a, 'CO2')
    mspt = get_series(a, 'MS_PT', 'MSPT')
    ev   = get_series(s, 'EV')

    return {
        't': t,
        's': s,
        'a': a,
        'COWN': cown,
        'CONG': cong,
        'CO2': co2,
        'MSPT': mspt,
        'EV': ev,
    }

results = {}
rows = []

for name, overrides in SCENARIOS.items():
    try:
        res = run_scenario(name, overrides)
        results[name] = res

        rows.append({
            'Сценарий': name,
            'COWN_2050': round(float(res['COWN'][-1]), 3) if res['COWN'] is not None else np.nan,
            'CONG_2050': round(float(res['CONG'][-1]), 3) if res['CONG'] is not None else np.nan,
            'CO2_2050': round(float(res['CO2'][-1]), 3) if res['CO2'] is not None else np.nan,
            'MSPT_2050': round(float(res['MSPT'][-1]), 3) if res['MSPT'] is not None else np.nan,
            'EV_2050': round(float(res['EV'][-1]), 3) if res['EV'] is not None else np.nan,
        })

    except Exception as e:
        print(f'ERROR in {name}: {type(e).__name__}: {e}')

df = pd.DataFrame(rows)
print('\n=== Таблица 2050 ===')
print(df.to_string(index=False))
df.to_csv('output/scenarios/scenarios_2050_fixed.csv', index=False)

# График
plot_vars = [
    ('COWN', 'COWN'),
    ('CONG', 'CONG'),
    ('CO2', 'CO2'),
    ('MSPT', 'MSPT'),
    ('EV', 'EV'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, (title, key) in zip(axes[:5], plot_vars):
    drawn = False
    for name in SCENARIOS:
        if name not in results:
            continue
        y = results[name].get(key)
        if y is None:
            continue
        ax.plot(YEARS[:len(y)], y, lw=2.4, color=COLORS[name], label=name)
        drawn = True
    ax.set_title(title)
    ax.set_xlabel('Год')
    if drawn:
        ax.legend(fontsize=8)

axes[5].axis('off')
plt.tight_layout()
plt.savefig('output/scenarios/fig_scenarios_trajectories_fixed.png', dpi=200, bbox_inches='tight')
plt.show()

print('\nSaved:')
print('  output/scenarios/scenarios_2050_fixed.csv')
print('  output/scenarios/fig_scenarios_trajectories_fixed.png')
