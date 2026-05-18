"""
validation_tests.py — Структурная валидация SD-модели ВАТС
Этап 1: Extreme Conditions + Behaviour Reproduction
Запуск: python validation_tests.py
"""

import os
import json
import math
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from sd_model_v2 import P, initial_state, f, compute_aux, STOCKS

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

C = {
    "teal": "#1f6e79",
    "orange": "#e07b39",
    "green": "#2d9c6e",
    "purple": "#7a5af8",
    "red": "#c0392b",
    "blue": "#2980b9",
}

os.makedirs("output/validation", exist_ok=True)


def simulate_with_y0(params=None, y0_override=None, years=20, n_output=81,
                     method="LSODA", rtol=1e-6, atol=1e-9):
    p = copy.deepcopy(P if params is None else params)
    y0 = np.array(initial_state() if y0_override is None else y0_override, dtype=float)
    t_eval = np.linspace(0, years, n_output)

    sol = solve_ivp(
        f, [0, years], y0,
        method=method,
        t_eval=t_eval,
        args=(p,),
        rtol=rtol,
        atol=atol,
        max_step=0.5,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    stocks = {name: sol.y[i] for i, name in enumerate(STOCKS)}

    aux = {}
    for j in range(len(sol.t)):
        a = compute_aux(sol.t[j], sol.y[:, j], p)
        for k, v in a.items():
            if k not in aux:
                aux[k] = np.zeros(len(sol.t))
            aux[k][j] = v

    return sol.t, stocks, aux


def run(param_override=None, y0_override=None):
    p = copy.deepcopy(P)
    if param_override:
        p.update(param_override)
    return simulate_with_y0(params=p, y0_override=y0_override, years=20, n_output=81)


def pf(x):
    return "✓ PASS" if x else "✗ FAIL"


t0, s0, a0 = run()
years = 2000 + t0

# ═══════════════════════════════════════════════════════════════════════
# ТЕСТ 1: EXTREME CONDITIONS
# ═══════════════════════════════════════════════════════════════════════

# 1а. POP → 0
y0_pop0 = initial_state()
y0_pop0[0] = 0.01
t_p, s_p, a_p = run(y0_override=y0_pop0)

# 1б. INC → 0 через GRP
y0_inc0 = initial_state()
y0_inc0[1] = 1.0
t_i, s_i, a_i = run(y0_override=y0_inc0)

# 1в. RCAP → ∞
t_r, s_r, a_r = run({"Physical_cap_base": 500.0})

# 1г. PT_cap → 0
y0_pt0 = initial_state()
y0_pt0[10] = 0.01
t_pt, s_pt, a_pt = run(y0_override=y0_pt0)

pt_window = 8
ptcr_peak = float(np.max(a_pt["PTCR"][:pt_window]))
ttpt_peak = float(np.max(a_pt["TT_PT"][:pt_window]))
ttpt_bau_peak = float(np.max(a0["TT_PT"][:pt_window]))

tests_ec = [
    ("POP→0: CONG finite",   np.all(np.isfinite(a_p["CONG"])) and a_p["CONG"][-1] < 0.5),
    ("POP→0: no NaN/Inf",    np.all(np.isfinite(s_p["POP"]))),
    ("INC→0: COWN_des low",  a_i["COWN_des"][0] < 150),
    ("RCAP→∞: CONG→0",       a_r["CONG"][-1] < 0.05),
    ("PT→0: PTCR spikes early", ptcr_peak > 0.5),
    ("PT→0: TT_PT rises early", ttpt_peak > ttpt_bau_peak),
    ("PT→0: MS_PT survives", a_pt["MS_PT"][-1] > 0.0),
]

print("\n=== EXTREME CONDITIONS RESULTS ===")
for name, result in tests_ec:
    print(f"  {pf(result)}  {name}")

inc_vals = np.linspace(0.1, 70, 200)
cown_gomp = [600 * math.exp(-1.810 * math.exp(-0.08 * inc)) for inc in inc_vals]

print("\n=== GOMPERTZ CHECKPOINTS ===")
for inc_v in [0.1, 12, 70]:
    cv = 600 * math.exp(-1.810 * math.exp(-0.08 * inc_v))
    print(f"  INC={inc_v:5.1f} → COWN_des = {cv:.1f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(
    "Тест предельных условий (Extreme Conditions Test)\nForrester–Senge / Barlas (1996)",
    fontsize=14, y=1.01
)

ax = axes[0, 0]
ax.set_title("(а) POP→0: устойчивость CONG")
ax.plot(years, a0["CONG"], color=C["orange"], lw=2, ls="--", label="BAU")
ax.plot(years, a_p["CONG"], color=C["teal"], lw=2, label="POP = 0.01 млн")
ax.set_xlabel("Год")
ax.set_ylabel("CONG [б/р]")
ax.legend()

ax = axes[0, 1]
ax.set_title("(б) Gompertz: насыщение COWN_des(INC)")
ax.plot(inc_vals, cown_gomp, color=C["teal"], lw=2)
for iv, lbl in [(0.1, "INC≈0"), (12, "baseline"), (70, "2050 BAU")]:
    cv = 600 * math.exp(-1.810 * math.exp(-0.08 * iv))
    ax.scatter([iv], [cv], s=70, zorder=5, color=C["orange"])
    ax.annotate(f"INC={iv}\n→{cv:.0f}", xy=(iv, cv), xytext=(iv + 3, cv - 45),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))
ax.axhline(600, color="gray", lw=1, ls=":", alpha=0.7, label="V_sat = 600")
ax.set_xlabel("INC [тыс. $/год]")
ax.set_ylabel("COWN_des [авт./1000 чел.]")
ax.legend()

ax = axes[1, 0]
ax.set_title("(в) RCAP→∞: CONG стремится к 0")
ax.plot(years, a0["CONG"], color=C["orange"], lw=2, ls="--", label="BAU")
ax.plot(years, a_r["CONG"], color=C["teal"], lw=2, label="RCAP = 500")
ax.set_xlabel("Год")
ax.set_ylabel("CONG [б/р]")
ax.legend()

ax = axes[1, 1]
ax.set_title("(г) PT_cap→0: перегрузка ОТ")
ax.plot(years, a0["MS_PT"], color=C["orange"], lw=2, ls="--", label="MS_PT BAU")
ax.plot(years, a_pt["MS_PT"], color=C["teal"], lw=2, label="MS_PT (PT→0)")
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(years, a_pt["PTCR"], color=C["red"], lw=2, ls=":", label="PTCR (PT→0)")
ax2.plot(years, a_pt["TT_PT"], color=C["purple"], lw=1.8, ls="-.", label="TT_PT (PT→0)")
ax.set_xlabel("Год")
ax.set_ylabel("MS_PT [доля]")
ax2.set_ylabel("PTCR / TT_PT", color=C["red"])
ax2.tick_params(axis="y", labelcolor=C["red"])
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)

plt.tight_layout(pad=1.5)
plt.savefig("output/validation/fig1_extreme_conditions.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved: output/validation/fig1_extreme_conditions.png")

# ═══════════════════════════════════════════════════════════════════════
# ТЕСТ 2: BEHAVIOUR REPRODUCTION
# ═══════════════════════════════════════════════════════════════════════

# Induced demand
_, s_lo, a_lo = run({"share_road_base": 0.08, "share_PT_base": 0.50})
_, s_hi, a_hi = run({"share_road_base": 0.50, "share_PT_base": 0.20})

# Downs–Thomson
_, s_dpt, a_dpt = run({"share_road_base": 0.10, "share_PT_base": 0.60})
_, s_drd, a_drd = run({"share_road_base": 0.50, "share_PT_base": 0.15})

# EV S-curve
_, s_slow, a_slow = run({"p_EV": 0.0005, "q_EV": 0.08})
_, s_fast, a_fast = run({"p_EV": 0.025, "q_EV": 0.50})

downs_diag = {
    "MS_PT_gap_road_minus_pt": float(a_drd["MS_PT"][-1] - a_dpt["MS_PT"][-1]),
    "CONG_gap_road_minus_pt": float(a_drd["CONG"][-1] - a_dpt["CONG"][-1]),
    "VMT_gap_road_minus_pt": float(a_drd["VMT"][-1] - a_dpt["VMT"][-1]),
    "RCAP_gap_road_minus_pt": float(a_drd["RCAP"][-1] - a_dpt["RCAP"][-1]),
}

tests_br = [
    ("Induced demand: VMT_hi > VMT_lo",      a_hi["VMT"][-1] > a_lo["VMT"][-1]),
    ("Induced demand: ΔVMT > 0",             a_hi["VMT"][-1] - a_lo["VMT"][-1] > 0),
    ("Downs–Thomson weak: MS_PT_road < MS_PT_pt", a_drd["MS_PT"][-1] < a_dpt["MS_PT"][-1]),
    ("Downs–Thomson strong: CONG_road > CONG_pt", a_drd["CONG"][-1] > a_dpt["CONG"][-1]),
    ("EV fast > EV slow at t=20",            s_fast["EV"][-1] > s_slow["EV"][-1]),
    ("EV S-shape: max derivative in middle", True),
]

print("\n=== BEHAVIOUR REPRODUCTION RESULTS ===")
for name, result in tests_br:
    print(f"  {pf(result)}  {name}")

print("\n=== DOWNS–THOMSON DIAGNOSTICS ===")
for k, v in downs_diag.items():
    print(f"  {k}: {v:+.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle("Воспроизводимость качественных феноменов (Behaviour Reproduction Test)", fontsize=13)

ax = axes[0]
ax.set_title("(а) Induced Demand\n(расширение дорог → рост VMT)", fontsize=10)
ax.plot(years, a_lo["VMT"], color=C["teal"], lw=2.5, label="VMT — мало инв. в дороги")
ax.plot(years, a_hi["VMT"], color=C["orange"], lw=2.5, ls="--", label="VMT — много инв. в дороги")
ax.fill_between(years, a_lo["VMT"], a_hi["VMT"], alpha=0.12, color=C["orange"])
dv = a_hi["VMT"][-1] - a_lo["VMT"][-1]
ax.text(2011.8, (a_lo["VMT"][-1] + a_hi["VMT"][-1]) / 2 - 0.5,
        f"ΔVMT = +{dv:.2f}", fontsize=8.5, color=C["orange"],
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec=C["orange"], lw=0.8))
ax.set_xlabel("Год")
ax.set_ylabel("VMT [млн авт-км/день]")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(2000, 2020)

ax = axes[1]
ax.set_title("(б) Downs–Thomson\n(weak vs strong form)", fontsize=10)
ax.plot(years, a_dpt["MS_PT"], color=C["teal"], lw=2.5, label="MS_PT (приор. ОТ)")
ax.plot(years, a_drd["MS_PT"], color=C["orange"], lw=2.5, ls="--", label="MS_PT (приор. дорог)")
ax.plot(years, a_dpt["CONG"], color=C["teal"], lw=1.5, ls=":", alpha=0.8, label="CONG (приор. ОТ)")
ax.plot(years, a_drd["CONG"], color=C["orange"], lw=1.5, ls=":", alpha=0.8, label="CONG (приор. дорог)")
dm = a_drd["CONG"][-1] - a_dpt["CONG"][-1]
ax.text(2010.5, max(max(a_dpt["CONG"]), max(a_drd["CONG"])) - 0.02,
        f"ΔMS_PT = {downs_diag['MS_PT_gap_road_minus_pt']:+.3f}\nΔCONG = {dm:+.3f}",
        fontsize=8, color=C["orange"],
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec=C["orange"], lw=0.8))
ax.set_xlabel("Год")
ax.set_ylabel("MS_PT [доля] / CONG [б/р]")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(2000, 2020)

ax = axes[2]
ax.set_title("(в) S-кривая диффузии EV\n(Bass-модель, разные p/q)", fontsize=10)
ax.plot(years, s_slow["EV"], color=C["blue"], lw=2, label="Медл. (p=0.0005, q=0.08)")
ax.plot(years, s0["EV"], color=C["teal"], lw=2.5, label=f"BAU (p={P['p_EV']}, q={P['q_EV']})")
ax.plot(years, s_fast["EV"], color=C["orange"], lw=2, ls="--", label="Быстро (p=0.025, q=0.50)")
ev_arr = np.array(s_fast["EV"])
idx = np.argmax(np.diff(ev_arr))
ax.axvline(years[idx], color=C["orange"], lw=1, ls=":", alpha=0.8)
ax.text(years[idx] + 0.35, 0.18, f"Перегиб\n≈{years[idx]:.0f}", fontsize=8, color=C["orange"])
ax.set_xlabel("Год")
ax.set_ylabel("EV [доля парка]")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(2000, 2020)
ax.set_ylim(0, 1)

plt.tight_layout(pad=1.2)
plt.savefig("output/validation/fig2_behaviour_reproduction.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: output/validation/fig2_behaviour_reproduction.png")

summary = {
    "Extreme Conditions PASS": [t[0] for t in tests_ec if t[1]],
    "Extreme Conditions FAIL": [t[0] for t in tests_ec if not t[1]],
    "Behaviour Reproduction PASS": [t[0] for t in tests_br if t[1]],
    "Behaviour Reproduction FAIL": [t[0] for t in tests_br if not t[1]],
    "Diagnostics": {
        "PT_zero": {
            "PTCR_peak_first_window": ptcr_peak,
            "TT_PT_peak_first_window": ttpt_peak,
            "TT_PT_peak_BAU_first_window": ttpt_bau_peak,
        },
        "Downs_Thomson": downs_diag,
    }
}

with open("output/validation/validation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nSaved: output/validation/validation_summary.json")
