"""Synthetic store layout: departments -> aisles -> bays, and walking times between bays.

Geometry: the back room door is at (0, 0). Aisle a runs along y at x = front + a * spacing.
Bay b of an aisle sits at y = (b + 0.5) * bay_width. Cross aisles run along the front (y = 0)
and the back (y = L). Walking between two aisles goes around whichever end is shorter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import load_assumptions

DEPOT = -1  # bay id of the back room


@dataclass
class StoreLayout:
    store_id: str
    item_bay: pd.DataFrame          # item_id, dept_id, bay
    bay_xy: np.ndarray              # (n_bays + 1, 2); last row is the back room
    aisle_of_bay: np.ndarray        # (n_bays + 1,); back room = -1
    aisle_len: float
    walk_ft_per_min: float

    @property
    def n_bays(self) -> int:
        return len(self.bay_xy) - 1

    def _idx(self, bay: int) -> int:
        return self.n_bays if bay == DEPOT else bay

    def distance_ft(self, b1: int, b2: int) -> float:
        i, j = self._idx(b1), self._idx(b2)
        if i == j:
            return 0.0
        (x1, y1), (x2, y2) = self.bay_xy[i], self.bay_xy[j]
        if self.aisle_of_bay[i] == self.aisle_of_bay[j] and self.aisle_of_bay[i] >= 0:
            return abs(y1 - y2)
        L = self.aisle_len
        return abs(x1 - x2) + min(y1 + y2, 2 * L - y1 - y2)

    def travel_min(self, b1: int, b2: int) -> int:
        """Walking minutes, rounded up to whole minutes (0 inside the same bay)."""
        d = self.distance_ft(b1, b2)
        return 0 if d == 0 else math.ceil(d / self.walk_ft_per_min)


def build_layout(store_id: str, items: pd.DataFrame, seed: int | None = None) -> StoreLayout:
    """items: item_id, dept_id for every item the store carries."""
    a = load_assumptions()
    lay = a["layout"]
    base_seed = a["seed"] if seed is None else seed
    rng = np.random.default_rng([base_seed, sum(map(ord, store_id))])
    depts = sorted(items["dept_id"].unique())
    order = list(rng.permutation(depts))
    rows = []
    bay = 0
    for dept in order:
        dept_items = items.loc[items["dept_id"] == dept, "item_id"].sort_values().tolist()
        for k in range(0, len(dept_items), lay["items_per_bay"]):
            for it in dept_items[k:k + lay["items_per_bay"]]:
                rows.append((it, dept, bay))
            bay += 1
    item_bay = pd.DataFrame(rows, columns=["item_id", "dept_id", "bay"])
    n_bays = bay
    bpa = lay["bays_per_aisle"]
    aisle = np.arange(n_bays) // bpa
    pos = np.arange(n_bays) % bpa
    x = lay["front_aisle_ft"] + aisle * lay["aisle_spacing_ft"]
    y = (pos + 0.5) * lay["bay_width_ft"]
    xy = np.vstack([np.column_stack([x, y]), [0.0, 0.0]])
    aisle_of_bay = np.append(aisle, -1)
    return StoreLayout(store_id, item_bay, xy, aisle_of_bay,
                       aisle_len=bpa * lay["bay_width_ft"], walk_ft_per_min=lay["walk_ft_per_min"])
