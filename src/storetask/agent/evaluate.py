"""Score the planning agent on eval/scenarios.jsonl.

Per scenario
  tool_ok    planning: create_tasks and schedule_tasks both called; questions: the matching find_*
             tool called and no planning tools; refusals: no tool calls at all
  args_ok    planning: the last create_tasks (include, exclude_departments) and schedule_tasks
             (absent_associates) arguments equal gold after normalising defaults
  plan_ok    planning: final scheduled task set equals the gold run (same tools, gold arguments)
  answer_ok  questions: every gold number appears in the final answer
  success    all applicable checks pass
"""
from __future__ import annotations

import argparse
import json
import re
import time

from ..config import ROOT
from .graph import make_llm, run_agent
from .planner import Store

PLAN_KINDS = {"full_plan", "price_only", "exclude_dept", "absent", "combo"}
ALL = ["price", "restock", "reset"]


def gold_plan(st: Store, g: dict) -> dict:
    tasks = st.build_tasks(g["store"], g["date"], tuple(g["include"]), tuple(g["exclude"]))
    sch, info, summ, errs = st.schedule(g["store"], tasks, tuple(g["absent"]))
    return {"task_ids": sorted(t.task_id for t in tasks), "summary": summ}


def gold_numbers(st: Store, sc: dict) -> list[int]:
    g = sc["gold"]
    if sc["kind"] == "q_price":
        df = st.price_changes(g["store"], g["date"])
        return [int(df.n_labels.sum()) if len(df) else 0]
    if sc["kind"] == "q_restock":
        df = st.restock_needs(g["store"], g["date"])
        return [len(df), int((df.tier_rank == 0).sum())]
    df = st.resets_due(g["store"], g["date"])
    return [int(b) for b in df.bay]


def norm_include(v):
    return sorted(v) if v else ALL


def score(sc: dict, run: dict, st: Store) -> dict:
    names = [c["name"] for c in run["calls"]]
    g = sc["gold"]
    res = {"id": sc["id"], "kind": sc["kind"], "latency_s": run["latency_s"], "steps": run["steps"],
           "tokens_in": run["tokens_in"], "tokens_out": run["tokens_out"], "n_calls": len(names), "tools": names}
    if sc["kind"] in PLAN_KINDS:
        ct = [c for c in run["calls"] if c["name"] == "create_tasks"]
        stc = [c for c in run["calls"] if c["name"] == "schedule_tasks"]
        res["tool_ok"] = bool(ct and stc)
        if ct and stc:
            ca, sa = ct[-1]["args"], stc[-1]["args"]
            res["args_ok"] = (ca.get("store_id") == g["store"] and ca.get("date") == g["date"]
                              and norm_include(ca.get("include")) == sorted(g["include"])
                              and sorted(ca.get("exclude_departments") or []) == g["exclude"]
                              and sa.get("store_id") == g["store"] and sa.get("date") == g["date"]
                              and sorted(sa.get("absent_associates") or []) == g["absent"])
        else:
            res["args_ok"] = False
        gp = gold_plan(st, g)
        key = f"{g['store']}|{g['date']}"
        got = set(run["plan_task_ids"].get(key, []))
        want = set(gp["task_ids"])
        res["task_jaccard"] = round(len(got & want) / len(got | want), 4) if (got | want) else 1.0
        sched = run["schedules"].get(key)
        res["plan_ok"] = bool(sched) and got == want and res["args_ok"]
        res["schedule_violations"] = sched["violations"] if sched else None
        res["success"] = res["tool_ok"] and res["args_ok"] and res["plan_ok"]
    elif sc["kind"].startswith("q_"):
        res["tool_ok"] = g["tool"] in names and not ({"create_tasks", "schedule_tasks"} & set(names))
        nums = gold_numbers(st, sc)
        found = {int(x) for x in re.findall(r"\b\d+\b", run["answer"] or "")}
        res["gold_numbers"] = nums
        res["answer_ok"] = all(n in found for n in nums)
        res["success"] = res["tool_ok"] and res["answer_ok"]
    else:
        res["tool_ok"] = len(names) == 0
        res["success"] = res["tool_ok"]
    res["answer"] = (run["answer"] or "")[:400]
    res["calls"] = run["calls"]
    return res


def summarize(rows: list[dict]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None
    by_kind = {}
    for k in sorted({r["kind"] for r in rows}):
        rr = [r for r in rows if r["kind"] == k]
        by_kind[k] = {"n": len(rr), "success": mean([r["success"] for r in rr])}
    plan = [r for r in rows if r["kind"] in PLAN_KINDS]
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "n": len(rows), "success": mean([r["success"] for r in rows]),
        "tool_selection": mean([r["tool_ok"] for r in rows]),
        "planning_args": mean([r["args_ok"] for r in plan]),
        "planning_plan_match": mean([r["plan_ok"] for r in plan]),
        "planning_task_jaccard": mean([r["task_jaccard"] for r in plan]),
        "question_answer": mean([r.get("answer_ok") for r in rows if r["kind"].startswith("q_")]),
        "refusal": mean([r["success"] for r in rows if r["kind"] == "refuse"]),
        "schedule_violations": sum(r.get("schedule_violations") or 0 for r in plan),
        "latency_median_s": lat[len(lat) // 2], "latency_p90_s": lat[int(0.9 * (len(lat) - 1))],
        "tokens_in_mean": mean([r["tokens_in"] for r in rows]), "tokens_out_mean": mean([r["tokens_out"] for r in rows]),
        "by_kind": by_kind,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    scen = [json.loads(line) for line in open(ROOT / "eval" / "scenarios.jsonl")]
    if args.limit:
        scen = scen[::max(1, len(scen) // args.limit)][:args.limit]
    st = Store()
    llm = make_llm(args.provider, args.model, args.base_url)
    rows = []
    t0 = time.time()
    for sc in scen:
        try:
            run = run_agent(sc["request"], llm, st)
        except Exception as e:
            run = {"answer": f"ERROR {e}"[:300], "calls": [], "schedules": {}, "plan_task_ids": {}, "latency_s": 0,
                   "steps": 0, "tokens_in": 0, "tokens_out": 0}
        rows.append(score(sc, run, st))
        print(f"{sc['id']:16s} success={rows[-1]['success']} {rows[-1]['latency_s']}s", flush=True)
    summ = summarize(rows)
    summ.update({"model": args.model, "provider": args.provider, "wall_s": round(time.time() - t0, 1)})
    out = ROOT / "eval" / "results" / (args.out or args.model.replace(":", "_").replace("/", "_"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    with open(out / "summary.json", "w") as fh:
        json.dump(summ, fh, indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
