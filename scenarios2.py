import copy
import numpy as np
import pandas as pd
import os

from sd_model_v2 import simulate, P, initial_state

os.makedirs('output/scenarios', exist_ok=True)

BASE_YEAR = 2000
END_YEAR = 2050
SIM_YEARS = END_YEAR - BASE_YEAR
N_OUTPUT = 251

def get_series(dct, *keys):
    for k in keys:
        if k in dct and dct[k] is not None:
            arr = np.asarray(dct[k], dtype=float)
            if len(arr) > 0:
                return arr
    return None

def run_model(overrides=None):
    p = copy.deepcopy(P)
    if overrides:
        p.update(overrides)

    t, s, a = simulate(
        p,
        y0_override=initial_state(),
        years=SIM_YEARS,
        n_output=N_OUTPUT
    )

    return {
        'COWN': get_series(s, 'COWN'),
        'EV': get_series(s, 'EV'),
        'CONG': get_series(a, 'CONG'),
        'CO2': get_series(a, 'CO2'),
        'MSPT': get_series(a, 'MS_PT', 'MSPT'),
    }

# Базовый прогон
bau = run_model({})

bau_2050 = {
    'COWN': float(bau['COWN'][-1]),
    'CONG': float(bau['CONG'][-1]),
    'CO2': float(bau['CO2'][-1]),
    'MSPT': float(bau['MSPT'][-1]) if bau['MSPT'] is not None else np.nan,
    'EV': float(bau['EV'][-1]),
}

print("BAU 2050:", bau_2050)

# Сетка усиленных policy-сценариев
park_grid = [8, 12, 16, 20, 25]
delta_grid = [0.4, 0.8, 1.2, 1.6, 2.0]

rows = []

for park in park_grid:
    for delta in delta_grid:
        overrides = {
            'ParkCost': park,
            'delta_own': delta,
        }

        try:
            res = run_model(overrides)

            cown_2050 = float(res['COWN'][-1])
            cong_2050 = float(res['CONG'][-1])
            co2_2050  = float(res['CO2'][-1])
            mspt_2050 = float(res['MSPT'][-1]) if res['MSPT'] is not None else np.nan
            ev_2050   = float(res['EV'][-1])

            # Улучшение относительно BAU:
            # меньше COWN/CONG/CO2 = лучше, больше MSPT = лучше
            d_cown = bau_2050['COWN'] - cown_2050
            d_cong = bau_2050['CONG'] - cong_2050
            d_co2  = bau_2050['CO2'] - co2_2050
            d_mspt = mspt_2050 - bau_2050['MSPT'] if not np.isnan(mspt_2050) else np.nan

            # Простой score для отбора "выигрышных" policy-сценариев
            score = (
                1.0 * d_cown +
                120.0 * d_cong +
                15.0 * d_co2 +
                800.0 * d_mspt
            )

            rows.append({
                'ParkCost': park,
                'delta_own': delta,
                'COWN_2050': round(cown_2050, 3),
                'CONG_2050': round(cong_2050, 3),
                'CO2_2050': round(co2_2050, 3),
                'MSPT_2050': round(mspt_2050, 3) if not np.isnan(mspt_2050) else np.nan,
                'EV_2050': round(ev_2050, 3),
                'dCOWN_vs_BAU': round(d_cown, 3),
                'dCONG_vs_BAU': round(d_cong, 5),
                'dCO2_vs_BAU': round(d_co2, 5),
                'dMSPT_vs_BAU': round(d_mspt, 5) if not np.isnan(d_mspt) else np.nan,
                'score': round(score, 3),
            })

        except Exception as e:
            rows.append({
                'ParkCost': park,
                'delta_own': delta,
                'COWN_2050': np.nan,
                'CONG_2050': np.nan,
                'CO2_2050': np.nan,
                'MSPT_2050': np.nan,
                'EV_2050': np.nan,
                'dCOWN_vs_BAU': np.nan,
                'dCONG_vs_BAU': np.nan,
                'dCO2_vs_BAU': np.nan,
                'dMSPT_vs_BAU': np.nan,
                'score': np.nan,
            })
            print(f"ERROR: ParkCost={park}, delta_own={delta}: {e}")

df = pd.DataFrame(rows)
df = df.sort_values('score', ascending=False).reset_index(drop=True)

print("\nTOP-10 policy scenarios:")
print(df.head(10).to_string(index=False))

df.to_csv('output/scenarios/restrict_grid_search.csv', index=False)
print("\nSaved: output/scenarios/restrict_grid_search.csv")
