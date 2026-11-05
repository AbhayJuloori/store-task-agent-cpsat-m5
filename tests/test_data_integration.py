from pathlib import Path

import duckdb
import numpy as np
import pytest

from storetask import schedule as scheduling
from storetask.agent.planner import Store
from storetask.config import load_assumptions
from storetask.sim import SimState, apply_day, day_tasks, load_store
from storetask.tasks import sql_forecast

M5_DB = Path("data/work/m5.duckdb")
AGENT_DB = Path("data/work/agent.duckdb")


@pytest.mark.skipif(not M5_DB.exists(), reason="data/work/m5.duckdb is not available")
def test_sql_forecast_matches_numpy_store_forecast_for_ca1():
    cfg = load_assumptions()
    d_num = cfg["window"]["plan_start_d"] + 6
    lookback = cfg["window"]["forecast_lookback_days"]
    con = duckdb.connect(str(M5_DB), read_only=True)
    try:
        store = load_store(con, "CA_1")
        sql = sql_forecast(con, "CA_1", d_num, lookback)
    finally:
        con.close()

    assert sql["item_id"].tolist() == store.items
    np.testing.assert_allclose(sql["forecast"].to_numpy(), store.forecast(d_num, lookback))


def _task_record(task):
    return (
        task.task_id,
        task.kind,
        task.bay,
        task.duration,
        task.tier,
        task.weight,
        task.skill,
        task.deadline,
        task.due_d,
        task.units,
        tuple(task.items),
    )


@pytest.mark.skipif(
    not (M5_DB.exists() and AGENT_DB.exists()),
    reason="data/work/m5.duckdb and data/work/agent.duckdb are required",
)
def test_agent_store_tasks_match_fcfs_closed_loop_simulator_new_tasks():
    cfg = load_assumptions()
    start = cfg["window"]["plan_start_d"]
    target = start + 6  # 2016-05-01
    con = duckdb.connect(str(M5_DB), read_only=True)
    try:
        sd = load_store(con, "CA_1")
        staff = scheduling.staff_from_config(cfg["staff"]["shifts"])
        state = SimState(level=np.floor(cfg["shelf"]["start_fraction"] * sd.cap))
        for d_num in range(start, target):
            tasks = day_tasks(sd, d_num, state)
            schedule = scheduling.fcfs(tasks, staff, sd.layout)
            apply_day(sd, d_num, state, tasks, schedule)

        simulator_tasks = day_tasks(sd, target, SimState(level=state.level.copy()))
    finally:
        con.close()

    agent = Store(AGENT_DB)
    try:
        agent_tasks = agent.build_tasks("CA_1", "2016-05-01")
    finally:
        agent.con.close()

    assert sorted(map(_task_record, agent_tasks)) == sorted(map(_task_record, simulator_tasks))
