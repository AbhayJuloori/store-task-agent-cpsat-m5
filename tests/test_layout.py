import math

import numpy as np
import pytest

from storetask.config import load_assumptions
from storetask.layout import DEPOT, build_layout


def test_build_layout_assigns_every_item_once_and_fills_aisles(synthetic_items):
    layout = build_layout("TEST_STORE", synthetic_items, seed=11)
    cfg = load_assumptions()["layout"]

    assert len(layout.item_bay) == len(synthetic_items)
    assert layout.item_bay["item_id"].is_unique
    assert set(layout.item_bay["item_id"]) == set(synthetic_items["item_id"])
    np.testing.assert_array_equal(
        layout.aisle_of_bay[:-1],
        np.arange(layout.n_bays) // cfg["bays_per_aisle"],
    )
    aisle_counts = np.bincount(layout.aisle_of_bay[:-1])
    assert np.all(aisle_counts <= cfg["bays_per_aisle"])
    assert np.all(aisle_counts[:-1] == cfg["bays_per_aisle"])


def test_layout_distances_and_travel_rounding(layout):
    bays = [DEPOT, *range(layout.n_bays)]
    for b1 in bays:
        assert layout.distance_ft(b1, b1) == 0
        assert layout.travel_min(b1, b1) == 0
        for b2 in bays:
            distance = layout.distance_ft(b1, b2)
            assert distance == pytest.approx(layout.distance_ft(b2, b1))
            expected = 0 if distance == 0 else math.ceil(distance / layout.walk_ft_per_min)
            assert layout.travel_min(b1, b2) == expected

    for b1 in range(layout.n_bays):
        for b2 in range(layout.n_bays):
            if layout.aisle_of_bay[b1] == layout.aisle_of_bay[b2]:
                assert layout.distance_ft(b1, b2) == pytest.approx(
                    abs(layout.bay_xy[b1, 1] - layout.bay_xy[b2, 1])
                )

    for bay in range(layout.n_bays):
        x, y = layout.bay_xy[bay]
        assert layout.distance_ft(DEPOT, bay) == pytest.approx(x + y)


def test_layout_distance_obeys_triangle_inequality(layout):
    rng = np.random.default_rng(5)
    bays = np.array([DEPOT, *range(layout.n_bays)])
    for a, b, c in rng.choice(bays, size=(100, 3), replace=True):
        direct = layout.distance_ft(int(a), int(c))
        via_b = layout.distance_ft(int(a), int(b)) + layout.distance_ft(int(b), int(c))
        assert direct <= via_b + 1e-9
