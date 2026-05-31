"""
cross_city_test.py — Тест переносимости (cross-city transfer test) SD-модели ВАТС.

Идея (transfer scaling, классификация TRB 2014):
    1. Калибруем поведенческие параметры модели на ГОРОДЕ-ДОНОРЕ (по умолч. Шэньчжэнь).
    2. Фиксируем эти параметры и подставляем НАЧАЛЬНЫЕ УСЛОВИЯ города-РЕЦИПИЕНТА
       (по умолч. Сингапур) из реальных данных recipient-города.
    3. Прогоняем единую структуру модели на рецепиенте БЕЗ повторной калибровки
       поведенческих параметров и считаем NRMSE относительно реальных рядов рецепиента.
    4. Высокий NRMSE (> ~25%) — ожидаемый и информативный результат: он количественно
       показывает, что унифицированная структура переносима по форме, но не по
       значениям без повторной подгонки (ср. MARS — рекалибруется в каждом городе).

Это НЕ "наивный перенос" (naive transfer): начальные условия берутся из реальных
данных recipient-города, а не копируются у донора.

Запуск:
    python cross_city_test.py                          # Шэньчжэнь -> Сингапур, быстрый режим
    python cross_city_test.py --donor singapore --recipient shenzhen
    python cross_city_test.py --maxiter 200 --popsize 15   # полная калибровка как в дипломе

Зависит от calibration.py (переиспользует загрузку данных, NRMSE, симуляцию).
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from sd_model_v2 import P

import calibration as cal


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default="shenzhen", choices=["shenzhen", "singapore"])
    ap.add_argument("--recipient", default="singapore", choices=["shenzhen", "singapore"])
    # быстрый, детерминированный режим по умолчанию (для воспроизводимости в репозитории);
    # для полной калибровки уровня диплома задайте --maxiter 200 --popsize 15
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    args, _ = ap.parse_known_args()
    if args.donor == args.recipient:
        ap.error("--donor и --recipient должны быть разными городами")
    return args


def calibrate_on(city, popsize, maxiter, seed, workers):
    """Калибрует поведенческие параметры на city, возвращает (p_opt, param_names, donor_metrics)."""
    df = cal.load_data(city)
    available = cal.build_available_series(df)
    if not available:
        raise ValueError(f"Нет пригодных рядов для {city}")

    train_mask = (df["year"] <= cal.TRAIN_END_YEAR).values
    test_mask = (df["year"] > cal.TRAIN_END_YEAR).values
    y0_city = cal.make_city_y0(city)

    calib_params = cal.get_calib_params(city)
    param_names = list(calib_params.keys())
    bounds = [(v[1], v[2]) for v in calib_params.values()]

    loss_fn = cal.build_loss_fn(
        df=df, available=available, train_mask=train_mask,
        y0_city=y0_city, param_names=param_names,
    )

    print(f"\n=== КАЛИБРОВКА НА ДОНОРЕ: {city.upper()} ===")
    print("Ряды:", list(available.keys()))
    print(f"DE: popsize={popsize}, maxiter={maxiter}, seed={seed}")

    t0 = time.time()
    result = differential_evolution(
        func=loss_fn, bounds=bounds,
        popsize=popsize, maxiter=maxiter, tol=1e-5,
        seed=seed, workers=workers,
        updating="deferred" if workers != 1 else "immediate",
        polish=True, init="sobol", mutation=(0.5, 1.5), recombination=0.7,
    )
    theta_opt = result.x
    p_opt = cal.theta_to_params(theta_opt, param_names)

    # NRMSE на доноре (для отчёта)
    bundle = cal.simulate_bundle(p_opt, y0_city=y0_city, df_years=df["year"].values)
    donor_metrics = cal.make_metrics_df(available, bundle, df, train_mask, test_mask)

    print(f"Калибровка завершена за {time.time()-t0:.1f} c, loss={result.fun:.4f}")
    print("\nNRMSE на доноре:")
    print(donor_metrics.to_string(index=False))

    return p_opt, param_names, theta_opt, donor_metrics


def evaluate_transfer(p_donor, recipient):
    """Прогоняет параметры донора на ВСЕХ годах рецепиента (с его Y0) и считает NRMSE."""
    df = cal.load_data(recipient)
    available = cal.build_available_series(df)
    if not available:
        raise ValueError(f"Нет пригодных рядов для {recipient}")

    # КЛЮЧЕВОЙ момент transfer scaling: параметры донора + начальные условия рецепиента
    y0_recipient = cal.make_city_y0(recipient)

    bundle = cal.simulate_bundle(p_donor, y0_city=y0_recipient, df_years=df["year"].values)

    rows = []
    nrmse_vals = []
    for model_key, (_, data_vals) in available.items():
        sim_vals = bundle["sim_at_data"].get(model_key)
        if sim_vals is None:
            continue
        obs = np.asarray(data_vals, dtype=float)
        # перенос оцениваем на ВСЁМ горизонте рецепиента (out-of-sample целиком)
        err = cal.nrmse(obs, np.asarray(sim_vals, dtype=float))
        rows.append({
            "Variable": model_key,
            "NRMSE_transfer_%": round(err * 100, 2) if pd.notna(err) else np.nan,
            "Pass_<25%": "✓" if pd.notna(err) and err < 0.25 else "✗",
        })
        if pd.notna(err):
            nrmse_vals.append(err)

    df_metrics = pd.DataFrame(rows)
    avg = float(np.mean(nrmse_vals)) if nrmse_vals else np.nan
    return df_metrics, avg, bundle


def main():
    args = parse_args()
    out_dir = os.path.join("output", "cross_city", f"{args.donor}_to_{args.recipient}")
    os.makedirs(out_dir, exist_ok=True)

    p_donor, param_names, theta_opt, donor_metrics = calibrate_on(
        args.donor, args.popsize, args.maxiter, args.seed, args.workers
    )

    print(f"\n=== ПЕРЕНОС: {args.donor.upper()} → {args.recipient.upper()} ===")
    print("Поведенческие параметры зафиксированы по донору,")
    print("начальные условия взяты из реальных данных рецепиента (transfer scaling).")

    df_metrics, avg_nrmse, _ = evaluate_transfer(p_donor, args.recipient)

    print("\nNRMSE переноса (на рецепиенте, весь горизонт):")
    print(df_metrics.to_string(index=False))
    print(f"\nСредний NRMSE переноса: {avg_nrmse*100:.1f}%")

    verdict = (
        "Перенос даёт высокий NRMSE (> 25%): структура переносима, "
        "но значения требуют повторной подгонки — что согласуется с\n"
        "практикой SD-LUTI (MARS рекалибруется в каждом городе)."
        if pd.notna(avg_nrmse) and avg_nrmse >= 0.25 else
        "Перенос даёт умеренный NRMSE (< 25%): города оказались достаточно близки."
    )
    print("\nВЫВОД:", verdict)

    summary = {
        "donor": args.donor,
        "recipient": args.recipient,
        "transfer_type": "transfer scaling (donor params + recipient initial conditions)",
        "de_settings": {"popsize": args.popsize, "maxiter": args.maxiter, "seed": args.seed},
        "donor_calibrated_params": {k: round(float(v), 6) for k, v in zip(param_names, theta_opt)},
        "donor_NRMSE": donor_metrics.to_dict(orient="records"),
        "transfer_NRMSE": df_metrics.to_dict(orient="records"),
        "transfer_NRMSE_avg_%": round(avg_nrmse * 100, 2) if pd.notna(avg_nrmse) else None,
    }
    with open(os.path.join(out_dir, "cross_city_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    df_metrics.to_csv(os.path.join(out_dir, "transfer_nrmse.csv"), index=False)

    print("\nSaved:")
    print(os.path.join(out_dir, "cross_city_summary.json"))
    print(os.path.join(out_dir, "transfer_nrmse.csv"))


if __name__ == "__main__":
    main()
