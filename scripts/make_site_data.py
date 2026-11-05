"""Builds docs/data/site.json (small, for the Pages charts) from the run report, the per-day
CSVs and the agent evaluation summaries."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs" / "data"

rep = json.loads((D / "run_report.json").read_text())
paired = pd.read_csv(D / "paired_store_days.csv")
closed = pd.read_csv(D / "closed_loop_store_days.csv")

daily = {}
for staffing, g in paired.groupby("staffing"):
    s = g.groupby(["d_num", "policy"])[["high_done", "high_total"]].sum()
    s = (s.high_done / s.high_total).unstack().round(4)
    daily[staffing] = {"d_num": s.index.tolist(), **{p: s[p].tolist() for p in s.columns}}

lost_daily = {}
for staffing, g in closed.groupby("staffing"):
    s = g.groupby(["d_num", "policy"])[["lost_units", "demand_units"]].sum()
    s = (s.lost_units / s.demand_units).unstack().round(4)
    lost_daily[staffing] = {"d_num": s.index.tolist(), **{p: s[p].tolist() for p in s.columns}}

agent = {}
res = ROOT / "eval" / "results"
for summ in sorted(res.glob("**/summary.json")):
    key = str(summ.parent.relative_to(res))
    agent[key] = json.loads(summ.read_text())

site = {"paired": rep["paired"], "closed_loop": rep["closed_loop"], "paired_differences": rep["paired_differences"],
        "solver": rep["solver"], "ablation": rep["ablation"], "checks": rep["checks"],
        "checks_passed": rep["checks_passed"], "runtime_s": rep["runtime_s"], "daily_high_coverage": daily,
        "daily_lost_share": lost_daily, "agent": agent}
(D / "site.json").write_text(json.dumps(site, indent=1))
print("wrote", D / "site.json", "agent runs:", list(agent))
