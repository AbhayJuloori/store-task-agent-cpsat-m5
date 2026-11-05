"""Build the small DuckDB file the agent queries.

Tables: calendar, departments, planogram (item -> bay -> aisle), sales_recent (units for the 28 days
before the plan window through its end), price_changes (plan window), shelf_snapshot (morning shelf
level and capacity per store, day and item, from the status-quo FCFS simulation), reset_calendar,
staff. All SQL task logic runs against this file, so it ships to Kaggle as one artifact.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .. import schedule as S
from ..config import DB_PATH, WORK, load_assumptions
from ..sim import SimState, apply_day, day_tasks, load_store
from ..tasks import reset_order

AGENT_DB = WORK / "agent.duckdb"
STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]


def build_agent_db(path=AGENT_DB, src=DB_PATH, stores=STORES) -> dict:
    a = load_assumptions()
    w = a["window"]
    p0, n = w["plan_start_d"], w["plan_days"]
    lb = w["forecast_lookback_days"]
    if path.exists():
        path.unlink()
    src_con = duckdb.connect(str(src), read_only=True)
    con = duckdb.connect(str(path))
    con.execute(f"ATTACH '{src}' AS m5 (READ_ONLY)")
    con.execute("CREATE TABLE calendar AS SELECT * FROM m5.calendar")
    con.execute("CREATE TABLE departments AS SELECT DISTINCT dept_id, cat_id FROM m5.items ORDER BY 1")
    con.execute(f"CREATE TABLE sales_recent AS SELECT * FROM m5.sales WHERE d_num BETWEEN {p0 - lb} AND {p0 + n - 1}")
    con.execute(f"""CREATE TABLE price_changes AS SELECT pc.* FROM m5.price_changes pc
                    JOIN m5.calendar c ON c.date = pc.effective_date WHERE c.d_num BETWEEN {p0} AND {p0 + n - 1}""")
    staff = S.staff_from_config(a["staff"]["shifts"])
    planos, snaps, resets = [], [], []
    for st in stores:
        sd = load_store(src_con, st)
        ib = sd.layout.item_bay.copy()
        ib["store_id"] = st
        ib["aisle"] = sd.layout.aisle_of_bay[ib["bay"].to_numpy()]
        planos.append(ib)
        order = reset_order(st, sd.layout.n_bays)
        k = a["resets"]["bays_per_day"]
        for i, d in enumerate(range(p0, p0 + n)):
            for b in order[i * k:(i + 1) * k]:
                resets.append((st, d, int(b)))
        state = SimState(level=np.floor(a["shelf"]["start_fraction"] * sd.cap))
        for d in range(p0, p0 + n):
            snaps.append(pd.DataFrame({"store_id": st, "d_num": d, "item_id": sd.items,
                                       "level": state.level.copy(), "capacity": sd.cap}))
            tasks = day_tasks(sd, d, state)
            apply_day(sd, d, state, tasks, S.fcfs(tasks, staff, sd.layout))
    planogram = pd.concat(planos, ignore_index=True)
    shelf = pd.concat(snaps, ignore_index=True)
    reset_df = pd.DataFrame(resets, columns=["store_id", "d_num", "bay"])
    staff_df = pd.DataFrame([{"associate_id": s.id, "shift_start_min": s.start, "shift_end_min": s.end,
                              "skills": ",".join(s.skills)} for s in staff])
    for name, df in [("planogram", planogram), ("shelf_snapshot", shelf), ("reset_calendar", reset_df),
                     ("staff", staff_df)]:
        con.register("tmp_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    con.execute("DETACH m5")
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ["calendar", "departments", "sales_recent", "price_changes", "planogram",
                        "shelf_snapshot", "reset_calendar", "staff"]}
    con.close()
    src_con.close()
    return counts


if __name__ == "__main__":
    print(build_agent_db())
