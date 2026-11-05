"""End-to-end run: load, simulate every store under each policy and staffing level, ablations,
consistency checks. Writes docs/data/run_report.json and docs/data/*.csv."""
from __future__ import annotations

import argparse
import json
import platform
import time

import duckdb
import numpy as np
import pandas as pd

from . import schedule as S
from .config import DB_PATH, DOCS_DATA, load_assumptions
from .load import build
from .sim import SimState, day_tasks, load_store, run_store, run_store_paired
from .tasks import sql_forecast, sql_price_changes

STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
STAFFING = {"4 associates": (), "3 associates (A4 out)": ("A4",)}


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def sql_consistency(con) -> dict:
    a = load_assumptions()
    sd = load_store(con, "CA_1")
    d = a["window"]["plan_start_d"] + 5
    f_sql = sql_forecast(con, "CA_1", d, a["window"]["forecast_lookback_days"])
    f_np = sd.forecast(d, a["window"]["forecast_lookback_days"])
    max_diff = float(np.abs(f_sql["forecast"].to_numpy() - f_np).max())
    pc_days = sorted(sd.price_items)
    pc_match = all(sql_price_changes(con, "CA_1", dd)["item_id"].tolist() == sd.price_items[dd] for dd in pc_days)
    return {"forecast_max_abs_diff": max_diff, "price_change_days": pc_days, "price_items_match": pc_match}


def ablation(con, stores, day: int, time_limit_s: float) -> list[dict]:
    """Exact-travel circuit model (minute grid and 15 minute grid) against the assignment model."""
    a = load_assumptions()
    rows = []
    for st in stores:
        sd = load_store(con, st)
        staff = S.staff_from_config(a["staff"]["shifts"], drop=("A4",))
        state = SimState(level=np.floor(a["shelf"]["start_fraction"] * sd.cap * 0.6))
        tasks = day_tasks(sd, day, state)
        g = S.greedy(tasks, staff, sd.layout)
        variants = {
            "greedy": (g, {}),
            "assignment (main)": S.cpsat(tasks, staff, sd.layout, time_limit_s=time_limit_s),
            "circuit, 1 min grid": S.cpsat_circuit(tasks, staff, sd.layout, slot=1, time_limit_s=time_limit_s),
            "circuit, 15 min grid": S.cpsat_circuit(tasks, staff, sd.layout, slot=15, time_limit_s=time_limit_s),
            "circuit, 1 min, greedy warm start": S.cpsat_circuit(tasks, staff, sd.layout, slot=1,
                                                                  time_limit_s=time_limit_s, hint=g),
        }
        for name, (sched, info) in variants.items():
            sm = S.summarize(tasks, staff, sched)
            rows.append({"store_id": st, "variant": name, "tasks": len(tasks), "high_total": sm["high_total"],
                         "high_done": sm["high_done"], "done": sm["done"], "travel_min": sm["travel_min"],
                         "violations": len(S.validate(tasks, staff, sd.layout, sched)),
                         "solve_s": info.get("solve_s"), "status": info.get("status")})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 stores x 7 days")
    ap.add_argument("--skip-load", action="store_true")
    args = ap.parse_args(argv)
    a = load_assumptions()
    t0 = time.time()
    report = {"config": a.raw, "python": platform.python_version()}
    if not args.skip_load or not DB_PATH.exists():
        report["load"] = build()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    stores = STORES[:2] if args.quick else STORES
    days = 7 if args.quick else a["window"]["plan_days"]

    closed, paired = [], []
    cache_dir = DB_PATH.parent / ("cache_quick" if args.quick else "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for li, (label, drop) in enumerate(STAFFING.items()):
        for st in stores:
            cpath = cache_dir / f"s{li}_{st}.pkl"
            if cpath.exists():
                c_df, p_df = pd.read_pickle(cpath)
            else:
                parts = []
                for pol in ("fcfs", "greedy", "cpsat"):
                    df = run_store(con, st, pol, days=days, staff_drop=drop)
                    df["staffing"] = label
                    parts.append(df)
                c_df = pd.concat(parts, ignore_index=True)
                p_df = run_store_paired(con, st, days=days, staff_drop=drop)
                p_df["staffing"] = label
                pd.to_pickle((c_df, p_df), cpath)
            closed.append(c_df)
            paired.append(p_df)
            print(f"{label} {st} done {time.time() - t0:.0f}s", flush=True)
    closed = pd.concat(closed, ignore_index=True)
    paired = pd.concat(paired, ignore_index=True)

    def agg(df):
        g = df.groupby(["staffing", "policy"])
        out = g[["tasks", "done", "high_total", "high_done", "medium_total", "medium_done", "travel_min",
                 "work_min", "demand_min", "shift_min", "weight_done"]].sum()
        out["high_coverage"] = out["high_done"] / out["high_total"]
        out["travel_per_task"] = out["travel_min"] / out["done"]
        out["load_ratio"] = out["demand_min"] / out["shift_min"]
        return out

    pa = agg(paired)
    ca = agg(closed)
    for col in ["lost_units", "demand_units", "price_labels_due_today", "price_labels_on_time"]:
        ca[col] = closed.groupby(["staffing", "policy"])[col].sum()
    ca["lost_share"] = ca["lost_units"] / ca["demand_units"]
    ca["backlog_reset_end"] = closed[closed.d_num == closed.d_num.max()].groupby(["staffing", "policy"])["backlog_reset"].sum()
    ca["backlog_price_end"] = closed[closed.d_num == closed.d_num.max()].groupby(["staffing", "policy"])["backlog_price"].sum()

    # paired differences per store-day
    diffs = {}
    for label in STAFFING:
        p = paired[paired.staffing == label].pivot_table(index=["store_id", "d_num"], columns="policy",
                                                          values=["high_done", "high_total", "travel_min", "done"])
        over = p[("high_total", "fcfs")] > 0
        cov = {pol: p[("high_done", pol)][over] / p[("high_total", pol)][over] for pol in ("fcfs", "greedy", "cpsat")}
        diffs[label] = {
            "store_days": int(len(p)),
            "cpsat_minus_fcfs_high_coverage": {"mean": round(float((cov["cpsat"] - cov["fcfs"]).mean()), 4),
                                               "ci95": boot_ci(cov["cpsat"] - cov["fcfs"])},
            "cpsat_minus_greedy_high_coverage": {"mean": round(float((cov["cpsat"] - cov["greedy"]).mean()), 4),
                                                 "ci95": boot_ci(cov["cpsat"] - cov["greedy"])},
            "cpsat_minus_fcfs_high_done": int((p[("high_done", "cpsat")] - p[("high_done", "fcfs")]).sum()),
            "cpsat_minus_greedy_high_done": int((p[("high_done", "cpsat")] - p[("high_done", "greedy")]).sum()),
            "days_cpsat_high_lt_greedy": int((p[("high_done", "cpsat")] < p[("high_done", "greedy")]).sum()),
            "days_cpsat_high_gt_greedy": int((p[("high_done", "cpsat")] > p[("high_done", "greedy")]).sum()),
        }
    cp = paired[paired.policy == "cpsat"]
    solve = {"median_s": round(float(cp.solver_solve_s.median()), 3),
             "p95_s": round(float(cp.solver_solve_s.quantile(0.95)), 3),
             "max_s": round(float(cp.solver_solve_s.max()), 3),
             "optimal_share": round(float((cp.solver_status == "OPTIMAL").mean()), 3),
             "tasks_median": int(cp.tasks.median()), "tasks_max": int(cp.tasks.max()),
             "repairs_total": int(cp.solver_repairs.sum()), "fills_total": int(cp.solver_fills.sum())}

    abl = ablation(con, stores[:3] if args.quick else STORES, a["window"]["plan_start_d"] + 5,
                   time_limit_s=10)
    print(f"ablation done {time.time() - t0:.0f}s", flush=True)
    sqlc = sql_consistency(con)

    checks = {
        "no_schedule_violations": int(closed.violations.sum() + paired.violations.sum()) == 0,
        "sql_forecast_equals_numpy": sqlc["forecast_max_abs_diff"] < 1e-9,
        "sql_price_changes_equal_sim": bool(sqlc["price_items_match"]),
        "paired_task_lists_identical": bool(paired.groupby(["staffing", "store_id", "d_num"]).tasks.nunique().max() == 1),
        "cpsat_never_below_fcfs_high_on_paired_days": bool(
            (paired.pivot_table(index=["staffing", "store_id", "d_num"], columns="policy", values="high_done")
             .eval("cpsat >= fcfs")).all()),
        "ablation_no_violations": all(r["violations"] == 0 for r in abl),
    }
    report.update({
        "stores": stores, "days": days, "runtime_s": round(time.time() - t0, 1),
        "paired": pa.reset_index().round(4).to_dict(orient="records"),
        "closed_loop": ca.reset_index().round(4).to_dict(orient="records"),
        "paired_differences": diffs, "solver": solve, "ablation": abl, "sql_consistency": sqlc,
        "checks": checks, "checks_passed": f"{sum(checks.values())}/{len(checks)}",
    })
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    name = "run_report_quick.json" if args.quick else "run_report.json"
    with open(DOCS_DATA / name, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    if not args.quick:
        paired.to_csv(DOCS_DATA / "paired_store_days.csv", index=False)
        closed.to_csv(DOCS_DATA / "closed_loop_store_days.csv", index=False)
    print(json.dumps({k: report[k] for k in ["checks_passed", "checks", "solver", "paired_differences", "runtime_s"]},
                     indent=1, default=str))
    return report


if __name__ == "__main__":
    main()
