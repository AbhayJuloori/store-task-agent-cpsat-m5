import numpy as np
import pandas as pd
import pytest

from storetask.layout import build_layout
from storetask.schedule import Assignment
from storetask.sim import CLOSE_MIN, OPEN_MIN, SimState, StoreData, apply_day, demand_share_before
from storetask.tasks import Task


def make_store(items, demand, capacity=None):
    meta = pd.DataFrame({"item_id": items, "dept_id": ["D"] * len(items)})
    return StoreData(
        store_id="S",
        items=items,
        sales=np.asarray(demand, dtype=float)[:, None],
        d0=100,
        cap=np.asarray(capacity or [10.0] * len(items)),
        layout=build_layout("S", meta, seed=3),
        price_items={},
    )


@pytest.mark.parametrize(
    ("minute", "expected"),
    [
        (OPEN_MIN - 1, 0.0),
        (OPEN_MIN, 0.0),
        ((OPEN_MIN + CLOSE_MIN) / 2, 0.5),
        (CLOSE_MIN, 1.0),
        (CLOSE_MIN + 1, 1.0),
    ],
)
def test_demand_share_before_boundaries(minute, expected):
    assert demand_share_before(minute) == expected


def test_apply_day_refills_at_finish_time_and_serves_covered_demand():
    sd = make_store(["restocked", "covered"], [6.0, 5.0])
    state = SimState(level=np.array([2.0, 8.0]))
    task = Task("R", "restock", 0, 10, "high", 1000, "R", None, 100, 8, ["restocked"])
    finish = (OPEN_MIN + CLOSE_MIN) // 2
    schedule = [Assignment("R", "A", finish - 10, finish, 0)]

    result = apply_day(sd, 100, state, [task], schedule)

    assert result["lost_units"] == pytest.approx(1.0)
    assert state.level[0] == pytest.approx(7.0)
    assert state.level[1] == pytest.approx(3.0)


def test_apply_day_counts_all_empty_shelf_demand_as_lost_without_restock():
    sd = make_store(["empty"], [4.0])
    state = SimState(level=np.array([0.0]))

    result = apply_day(sd, 100, state, [], [])

    assert result["lost_units"] == pytest.approx(4.0)
    assert result["demand_units"] == pytest.approx(4.0)
    assert state.level[0] == 0.0


def test_apply_day_has_no_lost_units_when_shelf_covers_demand():
    sd = make_store(["covered"], [5.0])
    state = SimState(level=np.array([8.0]))

    result = apply_day(sd, 100, state, [], [])

    assert result["lost_units"] == 0.0
    assert state.level[0] == pytest.approx(3.0)


def test_apply_day_backlogs_only_unfinished_price_and_reset_tasks():
    sd = make_store(["item"], [0.0])
    state = SimState(level=np.array([5.0]))
    tasks = [
        Task("P-done", "price", 0, 5, "high", 1000, "P", 720, 100, items=["item"]),
        Task("P-open", "price", 0, 5, "high", 1000, "P", 720, 100, items=["item"]),
        Task("S-open", "reset", 0, 5, "medium", 30, "S", None, 100),
        Task("R-open", "restock", 0, 5, "medium", 30, "R", None, 100, 1, ["item"]),
    ]
    schedule = [Assignment("P-done", "A", 400, 405, 1)]

    result = apply_day(sd, 100, state, tasks, schedule)

    assert [task.task_id for task in state.backlog] == ["P-open", "S-open"]
    assert result["backlog_price"] == 1
    assert result["backlog_reset"] == 1
    assert result["price_labels_due_today"] == 2
    assert result["price_labels_on_time"] == 1
