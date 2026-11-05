"""SQL task queries against agent.duckdb and the deterministic plan builder.

The agent never writes to the database. Its tools call these functions; the evaluation's gold
answers come from calling the same functions with the gold arguments.
"""
from __future__ import annotations

import math
from functools import lru_cache

import duckdb
import numpy as np

from .. import schedule as S
from ..config import load_assumptions
from ..layout import StoreLayout
from ..tasks import Task
from .data import AGENT_DB

RESTOCK_SQL = """
WITH fc AS (
    SELECT item_id, avg(units) AS forecast
    FROM sales_recent WHERE store_id = $store AND d_num BETWEEN $d - $lb AND $d - 1
    GROUP BY item_id
), s AS (
    SELECT sh.item_id, sh.level, sh.capacity, fc.forecast, p.bay, p.dept_id,
           CASE WHEN sh.level - fc.forecast < 0 AND fc.forecast > 0 THEN 0
                WHEN sh.level - fc.forecast < $reorder * sh.capacity THEN 1 END AS tier_rank
    FROM shelf_snapshot sh
    JOIN fc USING (item_id)
    JOIN planogram p ON p.store_id = sh.store_id AND p.item_id = sh.item_id
    WHERE sh.store_id = $store AND sh.d_num = $d
)
SELECT bay, any_value(dept_id) AS dept_id, min(tier_rank) AS tier_rank, count(*) AS n_items,
       CAST(sum(ceil(greatest(capacity - level, 0))) AS INTEGER) AS units,
       list(item_id ORDER BY item_id) AS items
FROM s WHERE tier_rank IS NOT NULL
GROUP BY bay ORDER BY bay
"""

PRICE_SQL = """
SELECT p.bay, any_value(p.dept_id) AS dept_id, count(*) AS n_labels, list(pc.item_id ORDER BY pc.item_id) AS items,
       round(avg(pc.new_price - pc.old_price), 3) AS mean_change
FROM price_changes pc
JOIN calendar c ON c.date = pc.effective_date
JOIN planogram p ON p.store_id = pc.store_id AND p.item_id = pc.item_id
WHERE pc.store_id = $store AND c.d_num = $d
GROUP BY p.bay ORDER BY p.bay
"""

RESET_SQL = """
SELECT r.bay, any_value(p.dept_id) AS dept_id
FROM reset_calendar r JOIN planogram p ON p.store_id = r.store_id AND p.bay = r.bay
WHERE r.store_id = $store AND r.d_num = $d GROUP BY r.bay ORDER BY r.bay
"""


class Store:
    def __init__(self, path=AGENT_DB):
        self.con = duckdb.connect(str(path), read_only=True)
        self.a = load_assumptions()

    # ---------------------------------------------------------- lookups
    def d_num(self, date: str) -> int:
        row = self.con.execute("SELECT d_num FROM calendar WHERE date = CAST(? AS DATE)", [date]).fetchone()
        if row is None:
            raise ValueError(f"unknown date {date}")
        d = row[0]
        w = self.a["window"]
        if not w["plan_start_d"] <= d < w["plan_start_d"] + w["plan_days"]:
            raise ValueError(f"{date} is outside the planning window 2016-04-25..2016-05-22")
        return d

    def day_info(self, store: str, date: str) -> dict:
        d = self.d_num(date)
        r = self.con.execute("SELECT weekday, wm_yr_wk, event_name_1 FROM calendar WHERE d_num = ?", [d]).fetchone()
        return {"store_id": store, "date": date, "weekday": r[0], "walmart_week": r[1], "event": r[2]}

    def departments(self) -> list[dict]:
        return self.con.execute("SELECT dept_id, cat_id FROM departments ORDER BY 1").df().to_dict("records")

    def staff(self) -> list[S.Associate]:
        return S.staff_from_config(self.a["staff"]["shifts"])

    @lru_cache(maxsize=16)
    def layout(self, store: str) -> StoreLayout:
        lay = self.a["layout"]
        df = self.con.execute("SELECT item_id, dept_id, bay FROM planogram WHERE store_id = ? ORDER BY bay, item_id",
                              [store]).df()
        n_bays = int(df.bay.max()) + 1
        bpa = lay["bays_per_aisle"]
        aisle = np.arange(n_bays) // bpa
        pos = np.arange(n_bays) % bpa
        xy = np.vstack([np.column_stack([lay["front_aisle_ft"] + aisle * lay["aisle_spacing_ft"],
                                         (pos + 0.5) * lay["bay_width_ft"]]), [0.0, 0.0]])
        return StoreLayout(store, df, xy, np.append(aisle, -1), bpa * lay["bay_width_ft"], lay["walk_ft_per_min"])

    def _q(self, sql: str, store: str, d: int):
        params = {"store": store, "d": d}
        if "$lb" in sql:
            params["lb"] = self.a["window"]["forecast_lookback_days"]
        if "$reorder" in sql:
            params["reorder"] = self.a["shelf"]["reorder_fraction"]
        return self.con.execute(sql, params).df()

    def restock_needs(self, store: str, date: str):
        return self._q(RESTOCK_SQL, store, self.d_num(date))

    def price_changes(self, store: str, date: str):
        return self._q(PRICE_SQL, store, self.d_num(date))

    def resets_due(self, store: str, date: str):
        return self._q(RESET_SQL, store, self.d_num(date))

    # ---------------------------------------------------------- plan
    def build_tasks(self, store: str, date: str, include=("price", "restock", "reset"),
                    exclude_departments=()) -> list[Task]:
        d = self.d_num(date)
        dur, w = self.a["durations"], self.a["priority"]["weights"]
        tiers = ("high", "medium", "low")
        ex = set(exclude_departments)
        tasks = []
        if "price" in include:
            for r in self.price_changes(store, date).itertuples():
                if r.dept_id in ex:
                    continue
                tasks.append(Task(f"{store}-d{d}-P-{r.bay}", "price", int(r.bay),
                                  math.ceil(dur["price_setup"] + dur["price_per_label"] * r.n_labels), "high",
                                  w["high"], "P", self.a["priority"]["price_deadline_min"], d, 0, list(r.items)))
        if "restock" in include:
            for r in self.restock_needs(store, date).itertuples():
                if r.dept_id in ex:
                    continue
                t = tiers[int(r.tier_rank)]
                tasks.append(Task(f"{store}-d{d}-R-{r.bay}", "restock", int(r.bay),
                                  math.ceil(dur["restock_setup"] + dur["restock_per_unit"] * r.units
                                            + dur["restock_per_item"] * r.n_items), t, w[t], "R", None, d,
                                  int(r.units), list(r.items)))
        if "reset" in include:
            for r in self.resets_due(store, date).itertuples():
                if r.dept_id in ex:
                    continue
                tasks.append(Task(f"{store}-d{d}-S-{r.bay}", "reset", int(r.bay), dur["reset_per_bay"], "medium",
                                  w["medium"], "S", None, d))
        return tasks

    def schedule(self, store: str, tasks: list[Task], absent=(), policy: str = "cpsat"):
        staff = [s for s in self.staff() if s.id not in set(absent)]
        lay = self.layout(store)
        if policy == "cpsat":
            sch, info = S.cpsat(tasks, staff, lay, time_limit_s=self.a["solver"]["time_limit_s"],
                                workers=self.a["solver"]["workers"], priority_scale=self.a["solver"]["priority_scale"])
        elif policy == "fcfs":
            sch, info = S.fcfs(tasks, staff, lay), {}
        else:
            sch, info = S.greedy(tasks, staff, lay), {}
        return sch, info, S.summarize(tasks, staff, sch), S.validate(tasks, staff, lay, sch)
