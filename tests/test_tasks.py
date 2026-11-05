import math

import numpy as np

from storetask.config import load_assumptions
from storetask.tasks import capacities, price_tasks, reset_order, reset_tasks, restock_tasks


def _items_by_bay(layout):
    return {
        int(bay): group["item_id"].tolist()
        for bay, group in layout.item_bay.groupby("bay", sort=True)
    }


def test_restock_tiers_formulas_and_grouping(layout):
    cfg = load_assumptions()
    by_bay = _items_by_bay(layout)
    bays = sorted(by_bay)
    high_item, grouped_medium = by_bay[bays[0]][:2]
    medium_item, boundary_item = by_bay[bays[1]][:2]
    items = layout.item_bay["item_id"].tolist()
    index = {item: i for i, item in enumerate(items)}
    cap = np.full(len(items), 10.0)
    level = cap.copy()
    forecast = np.zeros(len(items))

    level[index[high_item]], forecast[index[high_item]] = 1.0, 2.0
    level[index[grouped_medium]], forecast[index[grouped_medium]] = 4.0, 1.0
    level[index[medium_item]], forecast[index[medium_item]] = 4.0, 1.0
    level[index[boundary_item]], forecast[index[boundary_item]] = 4.5, 1.0

    tasks = restock_tasks("S", 100, layout, items, level, cap, forecast)
    assert [task.bay for task in tasks] == bays[:2]

    high, medium = tasks
    assert high.tier == "high"
    assert high.items == [high_item, grouped_medium]
    assert high.units == 15
    assert high.weight == cfg["priority"]["weights"]["high"]
    assert high.duration == math.ceil(
        cfg["durations"]["restock_setup"]
        + cfg["durations"]["restock_per_unit"] * high.units
        + cfg["durations"]["restock_per_item"] * len(high.items)
    )

    assert medium.tier == "medium"
    assert medium.items == [medium_item]
    assert boundary_item not in medium.items
    assert medium.units == 6
    assert medium.duration == math.ceil(
        cfg["durations"]["restock_setup"]
        + cfg["durations"]["restock_per_unit"] * medium.units
        + cfg["durations"]["restock_per_item"]
    )


def test_restock_emits_no_task_above_reorder_point(layout):
    items = layout.item_bay["item_id"].tolist()
    cap = np.full(len(items), 10.0)
    level = np.full(len(items), 8.0)
    forecast = np.full(len(items), 1.0)
    assert restock_tasks("S", 100, layout, items, level, cap, forecast) == []


def test_price_tasks_group_by_bay_with_required_attributes(layout):
    cfg = load_assumptions()
    by_bay = _items_by_bay(layout)
    bays = sorted(by_bay)
    changed = [*by_bay[bays[0]][:2], by_bay[bays[1]][0]]

    tasks = price_tasks("S", 101, layout, changed)

    assert [task.bay for task in tasks] == bays[:2]
    assert [len(task.items) for task in tasks] == [2, 1]
    for task in tasks:
        assert task.deadline == cfg["priority"]["price_deadline_min"]
        assert task.skill == "P"
        assert task.tier == "high"
        assert task.weight == cfg["priority"]["weights"]["high"]
        assert task.duration == math.ceil(
            cfg["durations"]["price_setup"]
            + cfg["durations"]["price_per_label"] * len(task.items)
        )


def test_reset_tasks_have_seeded_daily_order_and_count(layout):
    cfg = load_assumptions()
    start = 1914
    expected = reset_order("S", layout.n_bays)
    per_day = cfg["resets"]["bays_per_day"]

    first = reset_tasks("S", start, start, layout.n_bays)
    second = reset_tasks("S", start + 1, start, layout.n_bays)
    repeated = reset_tasks("S", start, start, layout.n_bays)

    assert len(first) == len(second) == per_day
    assert [task.bay for task in first] == expected[:per_day].tolist()
    assert [task.bay for task in second] == expected[per_day : 2 * per_day].tolist()
    assert [task.bay for task in repeated] == [task.bay for task in first]
    assert all(task.skill == "S" and task.tier == "medium" for task in first + second)


def test_capacities_apply_minimum_clamp_and_days_of_cover():
    cfg = load_assumptions()["shelf"]
    mean = np.array([0.0, 0.5, 2.1])
    expected = np.maximum(cfg["min_capacity"], np.ceil(cfg["days_of_cover"] * mean))
    np.testing.assert_array_equal(capacities(mean), expected)
