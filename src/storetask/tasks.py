"""Task generation from M5 data.

The SQL functions here are the read-only queries the agent's tools call. The simulator uses a
numpy copy of the same numbers for speed; tests check the two agree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from .config import load_assumptions
from .layout import StoreLayout

TIERS = ("high", "medium", "low")


@dataclass
class Task:
    task_id: str
    kind: str              # restock | price | reset
    bay: int
    duration: int          # minutes
    tier: str
    weight: int
    skill: str             # R | P | S
    deadline: int | None   # minute of the day the task must finish by, if any
    due_d: int             # day the task first became due
    units: int = 0
    items: list = field(default_factory=list)


# ---------------------------------------------------------------- SQL (agent tool layer)

def sql_forecast(con: duckdb.DuckDBPyConnection, store_id: str, d_num: int, lookback: int) -> pd.DataFrame:
    """Moving average of units over the `lookback` days strictly before d_num."""
    return con.execute("""
        SELECT item_id, avg(units) AS forecast
        FROM sales
        WHERE store_id = ? AND d_num BETWEEN ? AND ?
        GROUP BY item_id ORDER BY item_id
    """, [store_id, d_num - lookback, d_num - 1]).df()


def sql_price_changes(con: duckdb.DuckDBPyConnection, store_id: str, d_num: int) -> pd.DataFrame:
    """Items whose shelf price changes on calendar day d_num (first day of a Walmart week)."""
    return con.execute("""
        SELECT pc.item_id, pc.old_price, pc.new_price, pc.effective_date
        FROM price_changes pc
        JOIN calendar c ON c.date = pc.effective_date
        WHERE pc.store_id = ? AND c.d_num = ?
        ORDER BY pc.item_id
    """, [store_id, d_num]).df()


def sql_capacity_base(con: duckdb.DuckDBPyConnection, store_id: str, plan_start_d: int, days: int = 90) -> pd.DataFrame:
    return con.execute("""
        SELECT item_id, avg(units) AS mean_daily
        FROM sales WHERE store_id = ? AND d_num BETWEEN ? AND ?
        GROUP BY item_id ORDER BY item_id
    """, [store_id, plan_start_d - days, plan_start_d - 1]).df()


def sql_day_info(con: duckdb.DuckDBPyConnection, d_num: int) -> dict:
    row = con.execute("SELECT date, weekday, wm_yr_wk, event_name_1 FROM calendar WHERE d_num = ?",
                      [d_num]).fetchone()
    return {"date": str(row[0]), "weekday": row[1], "wm_yr_wk": row[2], "event": row[3]}


# ---------------------------------------------------------------- task builders

def capacities(mean_daily: np.ndarray) -> np.ndarray:
    s = load_assumptions()["shelf"]
    return np.maximum(s["min_capacity"], np.ceil(s["days_of_cover"] * mean_daily)).astype(float)


def restock_tasks(store_id: str, d_num: int, layout: StoreLayout, items: list[str],
                  level: np.ndarray, cap: np.ndarray, forecast: np.ndarray) -> list[Task]:
    a = load_assumptions()
    dur, w = a["durations"], a["priority"]["weights"]
    reorder = a["shelf"]["reorder_fraction"]
    proj = level - forecast
    tier = np.where((proj < 0) & (forecast > 0), 0, np.where(proj < reorder * cap, 1, -1))
    bay_of = layout.item_bay.set_index("item_id").loc[items, "bay"].to_numpy()
    tasks = []
    for bay in np.unique(bay_of[tier >= 0]):
        m = (bay_of == bay) & (tier >= 0)
        units = int(np.ceil((cap[m] - level[m]).clip(min=0)).sum())
        n_items = int(m.sum())
        t = TIERS[int(tier[m].min())]
        duration = math.ceil(dur["restock_setup"] + dur["restock_per_unit"] * units + dur["restock_per_item"] * n_items)
        tasks.append(Task(f"{store_id}-d{d_num}-R-{bay}", "restock", int(bay), duration, t, w[t], "R", None,
                          d_num, units, [items[i] for i in np.flatnonzero(m)]))
    return tasks


def price_tasks(store_id: str, d_num: int, layout: StoreLayout, changed_items: list[str]) -> list[Task]:
    a = load_assumptions()
    dur, w = a["durations"], a["priority"]["weights"]
    if not changed_items:
        return []
    ib = layout.item_bay.set_index("item_id").loc[changed_items, "bay"]
    tasks = []
    for bay, grp in ib.groupby(ib):
        n = len(grp)
        tasks.append(Task(f"{store_id}-d{d_num}-P-{bay}", "price", int(bay),
                          math.ceil(dur["price_setup"] + dur["price_per_label"] * n), "high", w["high"], "P",
                          a["priority"]["price_deadline_min"], d_num, 0, list(grp.index)))
    return tasks


def reset_order(store_id: str, n_bays: int) -> np.ndarray:
    a = load_assumptions()
    rng = np.random.default_rng([a["seed"], 7, sum(map(ord, store_id))])
    return rng.permutation(n_bays)


def reset_tasks(store_id: str, d_num: int, plan_start_d: int, n_bays: int) -> list[Task]:
    a = load_assumptions()
    k = a["resets"]["bays_per_day"]
    order = reset_order(store_id, n_bays)
    i0 = (d_num - plan_start_d) * k
    w = a["priority"]["weights"]["medium"]
    return [Task(f"{store_id}-d{d_num}-S-{int(b)}", "reset", int(b), a["durations"]["reset_per_bay"], "medium", w,
                 "S", None, d_num) for b in order[i0:i0 + k]]
