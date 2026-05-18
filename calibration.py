"""
calibration.py — калибровка SD-модели методом Differential Evolution

Запуск:
    python calibration.py --city shenzhen
    python calibration.py --city singapore
"""

import argparse
import copy
import json
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

from sd_model_v2 import simulate, P, initial_state


matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


BASE_YEAR = 2000
END_YEAR = 2020
TRAIN_END_YEAR = 2015
SIM_YEARS = END_YEAR - BASE_YEAR
SIM_N_OUTPUT = 201

CITY_Y0 = {
    "shenzhen": {"POP_0": 7.0, "GRP_0": 20.0, "COWN_0": 40.0},
    "singapore": {"POP_0": 4.0, "GRP_0": 90.0, "COWN_0": 100.0},
}

STATE_INDEX = {
    "POP_0": 0,
    "GRP_0": 1,
    "COWN_0": 5,
}

WEIGHTS = {
    "COWN": 1.5,
    "GRP": 1.0,
    "INC": 1.0,
    "POP": 0.5,
    "MS_PT": 1.2,
}

COLORS = {
    "teal": "#1f6e79",
    "orange": "#e07b39",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="shenzhen", choices=["shenzhen", "singapore"])
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)

    args, _ = parser.parse_known_args()
    return args


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_city_y0(city):
    y0 = initial_state()

    if isinstance(y0, dict):
        y0 = copy.deepcopy(y0)
        for key, val in CITY_Y0[city].items():
            state_name = key.replace("_0", "")
            if state_name in y0:
                y0[state_name] = val
        return y0

    arr = np.array(y0, dtype=float, copy=True)
    for key, val in CITY_Y0[city].items():
        idx = STATE_INDEX.get(key)
        if idx is not None and idx < len(arr):
            arr[idx] = val
    return arr


def to_calendar_years(t_sim):
    t_sim = np.asarray(t_sim, dtype=float)
    if t_sim.size == 0:
        return t_sim
    if np.nanmin(t_sim) >= -1e-12 and np.nanmax(t_sim) <= SIM_YEARS + 1e-12:
        return BASE_YEAR + t_sim
    return t_sim


def load_data(city):
    path = f"calibration_data/{city}_timeseries.csv"
    df = pd.read_csv(path)
    df = df.sort_values("year").reset_index(drop=True)

    if "year" not in df.columns:
        raise ValueError(f"В {path} нет колонки 'year'.")

    if "GRP_bln_CNY" in df.columns and "POP_mln" in df.columns:
        df["INC_calculated"] = df["GRP_bln_CNY"].astype(float) / df["POP_mln"].astype(float)

    if "MS_PT_pct" in df.columns:
        vals = pd.to_numeric(df["MS_PT_pct"], errors="coerce")
        if vals.notna().sum() > 0:
            if vals.max() > 1.5:
                df["MS_PT"] = vals / 100.0
            else:
                df["MS_PT"] = vals.astype(float)

    if "MS_PT" in df.columns:
        vals = pd.to_numeric(df["MS_PT"], errors="coerce")
        if vals.notna().sum() > 0 and vals.max() > 1.5:
            df["MS_PT"] = vals / 100.0

    return df


def build_available_series(df):
    candidates = {
        "COWN": ["COWN", "COWN_per_1000"],
        "GRP": ["GRP", "GRP_bln_CNY", "GRP_bln_USD"],
        "INC": ["INC", "GDPpercapita", "GDP_per_capita", "INC_calculated"],
        "POP": ["POP", "population", "POP_mln"],
        "MS_PT": ["MS_PT", "MSPT", "MS_PT_pct"],
    }

    available = {}
    for model_key, cols in candidates.items():
        for col in cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                if vals.notna().sum() >= 5:
                    if model_key == "MS_PT" and vals.max() > 1.5:
                        vals = vals / 100.0
                    available[model_key] = (col, vals.astype(float).values)
                    break
    return available


def get_calib_params(city):
    calib = {
        "V_sat_COWN": (P.get("V_sat_COWN", 600.0), 300.0, 900.0),
        "alpha_COWN": (P.get("alpha_COWN", -1.81), -3.0, -0.5),
        "beta_COWN": (P.get("beta_COWN", -0.08), -0.20, -0.02),
        "IncomeShare_base": (P.get("IncomeShare_base", 0.60), 0.40, 0.85),
        "alpha_GRP": (P.get("alpha_GRP", 0.06), 0.02, 0.15),
        "alpha_POP": (P.get("alpha_POP", 0.01), 0.005, 0.05),
        "Physical_cap_base": (P.get("Physical_cap_base", 18.0), 8.0, 40.0),
    }

    if city == "singapore" and "eta_COWN_DENS" in P:
        calib["eta_COWN_DENS"] = (P.get("eta_COWN_DENS", -0.30), -0.80, -0.05)

    return calib


def theta_to_params(theta, param_names):
    p = copy.deepcopy(P)
    for name, val in zip(param_names, theta):
        p[name] = float(val)
    return p


def nrmse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(sim)

    if valid.sum() < 2:
        return np.nan

    o = obs[valid]
    s = sim[valid]
    rmse = np.sqrt(np.mean((o - s) ** 2))
    rng = np.nanmax(o) - np.nanmin(o)

    if rng <= 1e-12:
        return rmse
    return rmse / rng


def run_simulation(p, y0_city, years=SIM_YEARS, n_output=SIM_N_OUTPUT):
    return simulate(p, years=years, n_output=n_output, y0_override=y0_city)


def extract_series(s_sim, a_sim):
    out = {}

    if "COWN" in s_sim:
        out["COWN"] = np.asarray(s_sim["COWN"], dtype=float)
    if "GRP" in s_sim:
        out["GRP"] = np.asarray(s_sim["GRP"], dtype=float)
    if "POP" in s_sim:
        out["POP"] = np.asarray(s_sim["POP"], dtype=float)
    if "INC" in a_sim:
        out["INC"] = np.asarray(a_sim["INC"], dtype=float)
    if "MS_PT" in a_sim:
        out["MS_PT"] = np.asarray(a_sim["MS_PT"], dtype=float)

    return out


def simulate_bundle(p, y0_city, df_years):
    t_sim, s_sim, a_sim = run_simulation(p, y0_city=y0_city, years=SIM_YEARS, n_output=SIM_N_OUTPUT)
    years_sim = to_calendar_years(t_sim)
    raw_series = extract_series(s_sim, a_sim)

    sim_at_data = {}
    for key, y_sim in raw_series.items():
        f = interp1d(years_sim, y_sim, bounds_error=False, fill_value="extrapolate")
        sim_at_data[key] = f(df_years)

    return {
        "t_sim": np.asarray(t_sim, dtype=float),
        "years_sim": np.asarray(years_sim, dtype=float),
        "s_sim": s_sim,
        "a_sim": a_sim,
        "raw_series": raw_series,
        "sim_at_data": sim_at_data,
    }


def build_loss_fn(df, available, train_mask, y0_city, param_names):
    def loss(theta):
        p = theta_to_params(theta, param_names)

        try:
            bundle = simulate_bundle(p, y0_city=y0_city, df_years=df["year"].values)
        except Exception:
            return 1e6

        total = 0.0
        count = 0.0

        for model_key, (_, data_vals) in available.items():
            sim_vals = bundle["sim_at_data"].get(model_key)
            if sim_vals is None:
                return 1e6

            obs_train = np.where(train_mask, data_vals.astype(float), np.nan)
            sim_train = np.where(train_mask, sim_vals.astype(float), np.nan)

            err = nrmse(obs_train, sim_train)
            if np.isnan(err):
                continue

            w = WEIGHTS.get(model_key, 1.0)
            total += w * err
            count += w

        try:
            s_chk = bundle["s_sim"]
            a_chk = bundle["a_sim"]
            penalty = 0.0

            if "COWN" in s_chk:
                cown = np.asarray(s_chk["COWN"], dtype=float)
                if np.any(~np.isfinite(cown)) or cown[-1] > 800 or cown[-1] < 10:
                    penalty += 10.0

            if "CONG" in a_chk:
                cong = np.asarray(a_chk["CONG"], dtype=float)
                if np.any(~np.isfinite(cong)) or cong[-1] > 5:
                    penalty += 10.0

            if "MS_PT" in a_chk:
                ms_pt = np.asarray(a_chk["MS_PT"], dtype=float)
                if np.any(~np.isfinite(ms_pt)) or np.any(ms_pt < 0) or np.any(ms_pt > 1):
                    penalty += 10.0

            total += penalty

        except Exception:
            return 1e6

        return total / count if count > 0 else 1e6

    return loss


def save_metrics(df_metrics, out_dir):
    df_metrics.to_csv(os.path.join(out_dir, "nrmse_metrics.csv"), index=False)


def save_params(df_params, out_dir):
    df_params.to_csv(os.path.join(out_dir, "calibrated_params.csv"), index=False)


def make_metrics_df(available, bundle_opt, df, train_mask, test_mask):
    rows = []

    for model_key, (_, data_vals) in available.items():
        sim_full = bundle_opt["sim_at_data"].get(model_key)
        if sim_full is None:
            continue

        obs_full = data_vals.astype(float)

        nrmse_train = nrmse(np.where(train_mask, obs_full, np.nan), np.where(train_mask, sim_full, np.nan))
        nrmse_test = nrmse(np.where(test_mask, obs_full, np.nan), np.where(test_mask, sim_full, np.nan))

        rows.append({
            "Variable": model_key,
            "NRMSE_train_%": round(nrmse_train * 100, 2) if pd.notna(nrmse_train) else np.nan,
            "NRMSE_test_%": round(nrmse_test * 100, 2) if pd.notna(nrmse_test) else np.nan,
            "Pass_test_<15%": "✓" if pd.notna(nrmse_test) and nrmse_test < 0.15 else "✗",
        })

    return pd.DataFrame(rows)


def make_params_df(calib_params, param_names, theta_opt, city):
    rows = []

    for pname, val_opt in zip(param_names, theta_opt):
        default = calib_params[pname][0]
        delta_pct = (val_opt - default) / abs(default) * 100 if abs(default) > 1e-12 else np.nan

        rows.append({
            "Parameter": pname,
            "Default": round(default, 4),
            f"Opt_{city}": round(float(val_opt), 4),
            "Delta_%": round(delta_pct, 1) if pd.notna(delta_pct) else np.nan,
        })

    return pd.DataFrame(rows)


def plot_real_vs_model(city, available, bundle_opt, df, train_mask, test_mask, df_metrics, out_dir):
    n_plots = len(available)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.6), squeeze=False)
    axes = axes.flatten()

    fig.suptitle(f"Calibration: {city.upper()} — observed vs model", fontsize=13)

    years_sim = bundle_opt["years_sim"]
    raw_series = bundle_opt["raw_series"]

    for ax, (model_key, (_, data_vals)) in zip(axes, available.items()):
        y_model = raw_series.get(model_key)
        obs = data_vals.astype(float)

        ax.set_title(model_key, fontsize=11)

        ax.plot(
            df["year"][train_mask],
            obs[train_mask],
            "o",
            color=COLORS["orange"],
            ms=6,
            label="Observed (train)",
            zorder=5,
        )
        ax.plot(
            df["year"][test_mask],
            obs[test_mask],
            "s",
            color=COLORS["teal"],
            ms=6,
            label="Observed (test)",
            zorder=5,
        )

        if y_model is not None:
            ax.plot(years_sim, y_model, color=COLORS["teal"], lw=2.5, label="Model")

        ax.axvline(TRAIN_END_YEAR, color="gray", ls=":", lw=1.2, alpha=0.7)

        row = df_metrics[df_metrics["Variable"] == model_key]
        if not row.empty:
            tr = row.iloc[0]["NRMSE_train_%"]
            te = row.iloc[0]["NRMSE_test_%"]
            ax.text(
                0.04,
                0.96,
                f"Train NRMSE: {tr}%\nTest NRMSE: {te}%",
                transform=ax.transAxes,
                va="top",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.7),
            )

        ax.set_xlabel("Year")
        ax.set_ylabel(model_key)
        ax.set_xlim(BASE_YEAR - 1, END_YEAR + 1)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(out_dir, "fig_real_vs_model.png"), dpi=200, bbox_inches="tight")
    plt.close()


def save_summary(city, result, theta_opt, param_names, elapsed, out_dir):
    summary = {
        "city": city,
        "optimizer": "differential_evolution",
        "train_period": f"{BASE_YEAR}–{TRAIN_END_YEAR}",
        "test_period": f"{TRAIN_END_YEAR + 1}–{END_YEAR}",
        "n_calib_params": len(param_names),
        "final_loss": float(result.fun),
        "n_iterations": int(result.nit),
        "n_evaluations": int(result.nfev),
        "elapsed_s": round(float(elapsed), 1),
        "optimized_params": {k: round(float(v), 6) for k, v in zip(param_names, theta_opt)},
    }

    with open(os.path.join(out_dir, "calibration_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    city = args.city
    out_dir = os.path.join("output", "calibration", city)
    ensure_dir(out_dir)

    df = load_data(city)
    available = build_available_series(df)

    if not available:
        raise ValueError("Не найдено пригодных рядов для калибровки.")

    print(f"\n=== CALIBRATION: {city.upper()} ===")
    print("Available series:", list(available.keys()))
    print("Years in CSV:", int(df['year'].min()), "–", int(df['year'].max()))

    train_mask = df["year"] <= TRAIN_END_YEAR
    test_mask = df["year"] > TRAIN_END_YEAR

    y0_city = make_city_y0(city)

    print("\nSmoke test simulate()...")
    t0, s0, a0 = run_simulation(P, y0_city=y0_city, years=SIM_YEARS, n_output=11)
    print("OK")
    print("State keys:", list(s0.keys())[:15])
    print("Aux keys:", list(a0.keys())[:20])

    calib_params = get_calib_params(city)
    param_names = list(calib_params.keys())
    bounds = [(v[1], v[2]) for v in calib_params.values()]

    print("\nParameters:")
    for name, (default, lo, hi) in calib_params.items():
        print(f"  {name}: default={default:.4f}, bounds=[{lo:.4f}, {hi:.4f}]")

    loss_fn = build_loss_fn(
        df=df,
        available=available,
        train_mask=train_mask.values,
        y0_city=y0_city,
        param_names=param_names,
    )

    counter = {"n": 0}

    def loss_with_counter(theta):
        counter["n"] += 1
        val = loss_fn(theta)
        if counter["n"] % 100 == 0:
            print(f"eval {counter['n']}: loss={val:.6f}")
        return val

    print(f"\nRunning Differential Evolution: popsize={args.popsize}, maxiter={args.maxiter}, workers={args.workers}")
    start = time.time()

    result = differential_evolution(
        func=loss_with_counter,
        bounds=bounds,
        popsize=args.popsize,
        maxiter=args.maxiter,
        tol=1e-5,
        seed=args.seed,
        workers=args.workers,
        updating="deferred" if args.workers != 1 else "immediate",
        polish=True,
        init="sobol",
        mutation=(0.5, 1.5),
        recombination=0.7,
    )

    elapsed = time.time() - start
    theta_opt = result.x
    p_opt = theta_to_params(theta_opt, param_names)
    bundle_opt = simulate_bundle(p_opt, y0_city=y0_city, df_years=df["year"].values)

    print(f"\nDone in {elapsed:.1f} s")
    print(f"Final loss: {result.fun:.6f}")

    df_metrics = make_metrics_df(
        available=available,
        bundle_opt=bundle_opt,
        df=df,
        train_mask=train_mask.values,
        test_mask=test_mask.values,
    )
    df_params = make_params_df(calib_params, param_names, theta_opt, city)

    print("\nNRMSE:")
    if not df_metrics.empty:
        print(df_metrics.to_string(index=False))

    print("\nOptimized parameters:")
    if not df_params.empty:
        print(df_params.to_string(index=False))

    save_metrics(df_metrics, out_dir)
    save_params(df_params, out_dir)
    plot_real_vs_model(city, available, bundle_opt, df, train_mask.values, test_mask.values, df_metrics, out_dir)
    save_summary(city, result, theta_opt, param_names, elapsed, out_dir)

    print("\nSaved:")
    print(os.path.join(out_dir, "nrmse_metrics.csv"))
    print(os.path.join(out_dir, "calibrated_params.csv"))
    print(os.path.join(out_dir, "fig_real_vs_model.png"))
    print(os.path.join(out_dir, "calibration_summary.json"))


if __name__ == "__main__":
    main()
