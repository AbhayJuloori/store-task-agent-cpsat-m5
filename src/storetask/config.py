from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "m5-forecasting-accuracy"
WORK = ROOT / "data" / "work"
DB_PATH = WORK / "m5.duckdb"
DOCS_DATA = ROOT / "docs" / "data"


@dataclass(frozen=True)
class Assumptions:
    raw: dict

    def __getitem__(self, key):
        return self.raw[key]


def load_assumptions(path: Path | None = None) -> Assumptions:
    path = path or ROOT / "configs" / "assumptions.yaml"
    with open(path) as fh:
        return Assumptions(yaml.safe_load(fh))
