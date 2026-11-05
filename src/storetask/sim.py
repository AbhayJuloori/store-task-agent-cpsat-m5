"""Closed-loop store simulation over the planning window.

Each morning: forecast (moving average of actual units before today), restock tasks from
shelf levels, price change tasks from real M5 sell_price changes, reset tasks from the reset
calendar, plus any unfinished price/reset backlog. A policy schedules the day. Completed
restocks fill their items to capacity at the task's finish time; actual M5 units then sell
uniformly over store hours. Lost units = demand that met an empty shelf.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from . import schedule as S
from .config import load_assumptions
from .layout import StoreLayout, build_layout
from .tasks import Task, capacities, price_tasks, reset_tasks, restock_tasks

OPEN_MIN, CLOSE_MIN = 360, 1380  # 06:00 to 23:00


def demand_share_before(minute: float) -> float:
    return float(np.clip((minute - OPEN_MIN) / (CLOSE_MIN - OPEN_MIN), 0.0, 1.0))


@dataclass
class StoreData:
    store_id: str
    items: list
    sales: np.ndarray            # (n_items, n_days) for d_num = d0 .. d1
    d0: int
    cap: np.ndarray
    layout: StoreLayout
    price_items: dict            # d_num -> list of item ids with a price change that day

    def units(self, d_num: int) -> np.ndarray:
        return self.sales[:, d_num - self.d0]

    def forecast(self, d_num: int, lookback: int) -> np.ndarray:
        j = d_num - self.d0
        return self.sales[:, j - lookback:j].mean(axis=1)


def load_store(con: duckdb.DuckDBPyConnection, store_id: str) -> StoreData:
    a = load_assumptions()
    w = a["window"]
    df = con.execute("""
        SELECT s.item_id, s.d_num, s.units FROM sales s WHERE store_id = ? ORDER BY item_id, d_num
    """, [store_id]).df()
    piv = df.pivot(index="item_id", columns="d_num", values="units").sort_index()
    items = piv.index.tolist()
    d0 = int(piv.columns.min())
    sales = piv.to_numpy(dtype=float)
    p0 = w["plan_start_d"]
    base = sales[:, p0 - 90 - d0:p0 - d0].mean(axis=1)
    meta = con.execute("SELECT item_id, dept_id FROM items ORDER BY item_id").df()
    meta = meta.set_index("item_id").loc[items].reset_index()
    layout = build_layout(store_id, meta)
    pc = con.execute("""
        SELECT c.d_num, pc.item_id FROM price_changes pc JOIN calendar c ON c.date = pc.effective_date
        WHERE pc.store_id = ? AND c.d_num BETWEEN ? AND ? ORDER BY 1, 2
    """, [store_id, p0, p0 + w["plan_days"] - 1]).df()
    price_items = {int(d): g["item_id"].tolist() for d, g in pc.groupby("d_num")}
    return StoreData(store_id, items, sales, d0, capacities(base), layout, price_items)


@dataclass
class SimState:
    level: np.ndarray
    backlog: list = field(default_factory=list)   # unfinished price / reset tasks


def day_tasks(sd: StoreData, d_num: int, state: SimState, include=("price", "restock", "reset"),
              exclude_depts: tuple = ()) -> list[Task]:
    a = load_assumptions()
    w = a["window"]
    fc = sd.forecast(d_num, w["forecast_lookback_days"])
    new = []
    if "price" in include:
        new += price_tasks(sd.store_id, d_num, sd.layout, sd.price_items.get(d_num, []))
    if "restock" in include:
        new += restock_tasks(sd.store_id, d_num, sd.layout, sd.items, state.level, sd.cap, fc)
    if "reset" in include:
        new += reset_tasks(sd.store_id, d_num, w["plan_start_d"], sd.layout.n_bays)
    carried = []
    for t in state.backlog:     # overdue tasks lose their same-day deadline but keep their tier
        carried.append(Task(t.task_id, t.kind, t.bay, t.duration, t.tier, t.weight, t.skill, None,
                            t.due_d, t.units, t.items))
    tasks = carried + new
    if exclude_depts:
        dept = sd.layout.item_bay.groupby("bay")["dept_id"].first()
        tasks = [t for t in tasks if dept[t.bay] not in exclude_depts]
    return tasks


def apply_day(sd: StoreData, d_num: int, state: SimState, tasks: list[Task], sched: list[S.Assignment]) -> dict:
    """Advance shelves through the day; return outcome metrics."""
    level = state.level.copy()
    demand = sd.units(d_num)
    finish = {s.task_id: s.end for s in sched}
    idx = {it: i for i, it in enumerate(sd.items)}
    fill_at = np.full(len(level), np.inf)
    for t in tasks:
        if t.kind == "restock" and t.task_id in finish:
            for it in t.items:
                fill_at[idx[it]] = finish[t.task_id]
    p = np.array([demand_share_before(f) if np.isfinite(f) else 1.0 for f in fill_at])
    served_before = np.minimum(level, demand * p)
    restocked = np.isfinite(fill_at)
    after_level = np.where(restocked, sd.cap, level - served_before)
    served_after = np.minimum(after_level, demand * (1 - p))
    lost = demand - served_before - served_after
    state.level = after_level - served_after
    price_done = [t for t in tasks if t.kind == "price" and t.task_id in finish]
    # same-day price tasks carry a deadline the schedulers enforce, so done today = on time
    price_on_time = sum(1 for t in price_done if t.due_d == d_num)
    state.backlog = [t for t in tasks if t.kind in ("price", "reset") and t.task_id not in finish]
    return {"lost_units": float(lost.sum()), "demand_units": float(demand.sum()),
            "price_labels_due_today": sum(len(t.items) for t in tasks if t.kind == "price" and t.due_d == d_num),
            "price_labels_on_time": sum(len(t.items) for t in price_done if t.due_d == d_num),
            "price_tasks_on_time": price_on_time,
            "backlog_price": sum(t.kind == "price" for t in state.backlog),
            "backlog_reset": sum(t.kind == "reset" for t in state.backlog)}


def run_store(con, store_id: str, policy: str, days: int | None = None, solver_kw: dict | None = None,
              staff_drop: tuple = ()) -> pd.DataFrame:
    a = load_assumptions()
    w = a["window"]
    sd = load_store(con, store_id)
    staff = S.staff_from_config(a["staff"]["shifts"], drop=staff_drop)
    state = SimState(level=np.floor(a["shelf"]["start_fraction"] * sd.cap))
    rows = []
    for d in range(w["plan_start_d"], w["plan_start_d"] + (days or w["plan_days"])):
        tasks = day_tasks(sd, d, state)
        sched, info = schedule_with(policy, tasks, staff, sd.layout, solver_kw)
        errs = S.validate(tasks, staff, sd.layout, sched)
        summ = S.summarize(tasks, staff, sched)
        out = apply_day(sd, d, state, tasks, sched)
        rows.append({"store_id": store_id, "d_num": d, "policy": policy, "violations": len(errs),
                     **summ, **out, **{f"solver_{k}": v for k, v in info.items()}})
    return pd.DataFrame(rows)


def schedule_with(policy: str, tasks, staff, layout, solver_kw: dict | None = None):
    a = load_assumptions()
    if policy == "fcfs":
        return S.fcfs(tasks, staff, layout), {}
    if policy == "greedy":
        return S.greedy(tasks, staff, layout), {}
    if policy == "cpsat":
        kw = dict(time_limit_s=a["solver"]["time_limit_s"], workers=a["solver"]["workers"],
                  priority_scale=a["solver"]["priority_scale"], travel_weight=a["solver"]["travel_weight"])
        kw.update(solver_kw or {})
        return S.cpsat(tasks, staff, layout, **kw)
    raise ValueError(policy)


def run_store_paired(con, store_id: str, policies=("fcfs", "greedy", "cpsat"), days: int | None = None,
                     staff_drop: tuple = (), state_policy: str = "fcfs") -> pd.DataFrame:
    """Open loop: every policy schedules the identical task list each day. Shelves advance with
    the `state_policy` schedule (the status quo), so day t's tasks do not depend on the policy
    under test."""
    a = load_assumptions()
    w = a["window"]
    sd = load_store(con, store_id)
    staff = S.staff_from_config(a["staff"]["shifts"], drop=staff_drop)
    state = SimState(level=np.floor(a["shelf"]["start_fraction"] * sd.cap))
    rows = []
    for d in range(w["plan_start_d"], w["plan_start_d"] + (days or w["plan_days"])):
        tasks = day_tasks(sd, d, state)
        keep = None
        for pol in policies:
            sched, info = schedule_with(pol, tasks, staff, sd.layout)
            errs = S.validate(tasks, staff, sd.layout, sched)
            summ = S.summarize(tasks, staff, sched)
            # timing quality: minute of day when each high tier restock finishes
            fin = {s.task_id: s.end for s in sched}
            hi = [fin[t.task_id] for t in tasks if t.tier == "high" and t.kind == "restock" and t.task_id in fin]
            rows.append({"store_id": store_id, "d_num": d, "policy": pol, "violations": len(errs),
                         "high_restock_mean_finish": float(np.mean(hi)) if hi else None, **summ,
                         **{f"solver_{k}": v for k, v in info.items()}})
            if pol == state_policy:
                keep = sched
        apply_day(sd, d, state, tasks, keep)
    return pd.DataFrame(rows)
