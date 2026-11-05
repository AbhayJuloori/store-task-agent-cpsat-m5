"""Assign and sequence a store day's tasks across associate shifts.

Policies
  fcfs     first come first served: tasks in queue order go to whichever skilled associate is
           free first; a task that no longer fits a shift (or its deadline) is skipped.
  greedy   priority then nearest: each associate, when free, takes the highest tier task it can
           finish, breaking ties by walking time. A stronger baseline than FCFS.
  cpsat    OR-Tools CP-SAT: one circuit per associate over the tasks it is skilled for, travel
           times on the arcs, shift windows, deadlines, start times on a slot grid; maximise
           priority weight done, then minimise walking.

Every schedule is checked by `validate`, which recomputes feasibility from scratch.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

from .layout import DEPOT, StoreLayout
from .tasks import TIERS, Task


@dataclass(frozen=True)
class Associate:
    id: str
    start: int
    end: int
    skills: tuple


@dataclass(frozen=True)
class Assignment:
    task_id: str
    associate: str
    start: int
    end: int
    travel: int  # walking minutes into this task


def staff_from_config(cfg: list[dict], drop: tuple = ()) -> list[Associate]:
    return [Associate(s["id"], s["start"], s["end"], tuple(s["skills"])) for s in cfg if s["id"] not in drop]


# ---------------------------------------------------------------- validation + metrics

def validate(tasks: list[Task], staff: list[Associate], layout: StoreLayout, sched: list[Assignment]) -> list[str]:
    """Return a list of violations (empty = feasible)."""
    errs = []
    tmap = {t.task_id: t for t in tasks}
    amap = {a.id: a for a in staff}
    seen = set()
    for s in sched:
        if s.task_id in seen:
            errs.append(f"{s.task_id} scheduled twice")
        seen.add(s.task_id)
    for aid in {s.associate for s in sched}:
        a = amap[aid]
        rows = sorted([s for s in sched if s.associate == aid], key=lambda s: s.start)
        loc, free = DEPOT, a.start
        for s in rows:
            t = tmap[s.task_id]
            if t.skill not in a.skills:
                errs.append(f"{aid} lacks skill {t.skill} for {t.task_id}")
            need = layout.travel_min(loc, t.bay)
            if s.start < free + need:
                errs.append(f"{t.task_id} starts {s.start} before {free}+{need}")
            if s.end != s.start + t.duration:
                errs.append(f"{t.task_id} wrong duration")
            if s.end > a.end:
                errs.append(f"{t.task_id} ends after shift")
            if t.deadline is not None and s.end > t.deadline:
                errs.append(f"{t.task_id} misses deadline")
            if s.travel != need:
                errs.append(f"{t.task_id} travel {s.travel} != {need}")
            loc, free = t.bay, s.end
    return errs


def summarize(tasks: list[Task], staff: list[Associate], sched: list[Assignment]) -> dict:
    done = {s.task_id for s in sched}
    out = {"tasks": len(tasks), "done": len(done), "travel_min": sum(s.travel for s in sched),
           "work_min": sum(t.duration for t in tasks if t.task_id in done),
           "demand_min": sum(t.duration for t in tasks),
           "shift_min": sum(a.end - a.start for a in staff),
           "weight_done": sum(t.weight for t in tasks if t.task_id in done),
           "weight_total": sum(t.weight for t in tasks)}
    for tier in TIERS:
        ts = [t for t in tasks if t.tier == tier]
        out[f"{tier}_total"] = len(ts)
        out[f"{tier}_done"] = sum(t.task_id in done for t in ts)
    return out


# ---------------------------------------------------------------- baselines

def _fits(t: Task, a: Associate, start: int) -> bool:
    end = start + t.duration
    return end <= a.end and (t.deadline is None or end <= t.deadline)


def fcfs(tasks: list[Task], staff: list[Associate], layout: StoreLayout) -> list[Assignment]:
    """Queue order = the order tasks were emitted (price, then restock by bay, then resets)."""
    state = {a.id: (a.start, DEPOT) for a in staff}
    sched = []
    for t in tasks:
        best = None
        for a in staff:
            if t.skill not in a.skills:
                continue
            free, loc = state[a.id]
            tr = layout.travel_min(loc, t.bay)
            st = free + tr
            if _fits(t, a, st) and (best is None or st < best[0]):
                best = (st, a, tr)
        if best:
            st, a, tr = best
            sched.append(Assignment(t.task_id, a.id, st, st + t.duration, tr))
            state[a.id] = (st + t.duration, t.bay)
    return sched


def greedy(tasks: list[Task], staff: list[Associate], layout: StoreLayout) -> list[Assignment]:
    rank = {tier: i for i, tier in enumerate(TIERS)}
    open_tasks = list(tasks)
    state = {a.id: (a.start, DEPOT) for a in staff}
    active = set(state)
    sched = []
    while active and open_tasks:
        aid = min(active, key=lambda k: (state[k][0], k))
        a = next(x for x in staff if x.id == aid)
        free, loc = state[aid]
        cands = []
        for t in open_tasks:
            if t.skill not in a.skills:
                continue
            tr = layout.travel_min(loc, t.bay)
            if _fits(t, a, free + tr):
                dl = t.deadline if t.deadline is not None else 10**6
                cands.append((rank[t.tier], dl, tr, t.task_id, t))
        if not cands:
            active.discard(aid)
            continue
        _, _, tr, _, t = min(cands, key=lambda c: c[:4])
        st = free + tr
        sched.append(Assignment(t.task_id, aid, st, st + t.duration, tr))
        state[aid] = (st + t.duration, t.bay)
        open_tasks.remove(t)
    return sched


# ---------------------------------------------------------------- CP-SAT

def cpsat_circuit(tasks: list[Task], staff: list[Associate], layout: StoreLayout, slot: int = 15,
          time_limit_s: float = 10.0, workers: int = 4, priority_scale: int = 10, travel_weight: int = 1,
          neighbors: int | None = 20, hint: list[Assignment] | None = None, seed: int = 0):
    """Return (schedule, info). `slot` = start-time grid in minutes (1 = exact minutes)."""
    t0 = time.time()
    m = cp_model.CpModel()
    n = len(tasks)
    horizon = max(a.end for a in staff)
    bays = [t.bay for t in tasks]
    tr = [[layout.travel_min(bays[i], bays[j]) for j in range(n)] for i in range(n)]
    tr0 = [layout.travel_min(DEPOT, b) for b in bays]

    start = []
    for t in tasks:
        k = m.NewIntVar(0, horizon // slot, f"k_{t.task_id}")
        s = m.NewIntVar(0, horizon, f"s_{t.task_id}")
        m.Add(s == slot * k)
        start.append(s)
    done = [m.NewBoolVar(f"done_{t.task_id}") for t in tasks]
    hint_next = {}
    hmap = {h.task_id: h for h in (hint or [])}
    idx_of = {t.task_id: i for i, t in enumerate(tasks)}
    if hint:
        for a in staff:
            seq = [idx_of[h.task_id] for h in sorted((h for h in hint if h.associate == a.id), key=lambda h: h.start)]
            for p_, q_ in zip(seq, seq[1:]):
                hint_next[a.id, p_] = q_
            if seq:
                hint_next[a.id, "depot"] = seq[0]
                hint_next[a.id, seq[-1]] = "depot"
    x = {}
    arcs_total = 0
    travel_terms = []
    for a in staff:
        elig = [i for i, t in enumerate(tasks) if t.skill in a.skills and a.start + tr0[i] + t.duration <= a.end
                and (t.deadline is None or a.start + tr0[i] + t.duration <= t.deadline)]
        for i in elig:
            x[a.id, i] = m.NewBoolVar(f"x_{a.id}_{i}")
        if not elig:
            continue
        allowed = {}
        for i in elig:
            others = sorted((j for j in elig if j != i), key=lambda j: (tr[i][j], j))
            if neighbors is not None:
                keep = set(others[:neighbors])
                if (a.id, i) in hint_next and hint_next[a.id, i] != "depot":
                    keep.add(hint_next[a.id, i])
                others = [j for j in others if j in keep]
            allowed[i] = others
        arcs = []
        node = {i: k + 1 for k, i in enumerate(elig)}  # 0 = depot
        idle = m.NewBoolVar(f"idle_{a.id}")
        arcs.append((0, 0, idle))
        hints = []
        if hint:
            hints.append((idle, int((a.id, "depot") not in hint_next)))
        for i in elig:
            arcs.append((node[i], node[i], x[a.id, i].Not()))
            lit = m.NewBoolVar("")
            arcs.append((0, node[i], lit))
            if hint:
                hints.append((lit, int(hint_next.get((a.id, "depot")) == i)))
            m.Add(start[i] >= a.start + tr0[i]).OnlyEnforceIf(lit)
            travel_terms.append(tr0[i] * lit)
            back = m.NewBoolVar("")
            arcs.append((node[i], 0, back))
            if hint:
                hints.append((back, int(hint_next.get((a.id, i)) == "depot")))
            m.Add(start[i] + tasks[i].duration <= a.end).OnlyEnforceIf(x[a.id, i])
            m.Add(start[i] >= a.start).OnlyEnforceIf(x[a.id, i])
            for j in allowed[i]:
                lit = m.NewBoolVar("")
                arcs.append((node[i], node[j], lit))
                if hint:
                    hints.append((lit, int(hint_next.get((a.id, i)) == j)))
                m.Add(start[j] >= start[i] + tasks[i].duration + tr[i][j]).OnlyEnforceIf(lit)
                if tr[i][j]:
                    travel_terms.append(tr[i][j] * lit)
        for lit, v in hints:
            m.AddHint(lit, v)
        arcs_total += len(arcs)
        m.AddCircuit(arcs)
        m.Add(sum(x[a.id, i] for i in elig) == 0).OnlyEnforceIf(idle)
    for i, t in enumerate(tasks):
        mine = [x[k] for k in x if k[1] == i]
        m.Add(sum(mine) == done[i]) if mine else m.Add(done[i] == 0)
        if t.deadline is not None:
            m.Add(start[i] + t.duration <= t.deadline).OnlyEnforceIf(done[i])
    m.Maximize(priority_scale * sum(t.weight * done[i] for i, t in enumerate(tasks))
               - travel_weight * sum(travel_terms))
    if hint:
        for i, t in enumerate(tasks):
            h = hmap.get(t.task_id)
            m.AddHint(done[i], 1 if h else 0)
            if h and h.start % slot == 0:
                m.AddHint(start[i], h.start)
            for a in staff:
                if (a.id, i) in x:
                    m.AddHint(x[a.id, i], 1 if h and h.associate == a.id else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    status = solver.Solve(m)
    info = {"status": solver.StatusName(status), "solve_s": round(time.time() - t0, 3),
            "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            "bound": solver.BestObjectiveBound(), "arcs": arcs_total, "n_tasks": n}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], info
    sched = []
    for a in staff:
        mine = sorted((solver.Value(start[i]), i) for (aid, i) in x if aid == a.id and solver.Value(x[aid, i]))
        loc = DEPOT
        for st, i in mine:
            t = tasks[i]
            sched.append(Assignment(t.task_id, a.id, st, st + t.duration, layout.travel_min(loc, t.bay)))
            loc = t.bay
    return sched, info


# ---------------------------------------------------------------- CP-SAT, assignment + route

def _route(tasks_a: list[Task], a: Associate, layout: StoreLayout) -> list[Task]:
    """Deadline tasks first, then the rest of the high tier, then everything else; each block is
    walked in serpentine order (aisles in order, bays alternating direction). High tier restocks
    go first because a shelf filled at 08:00 loses fewer sales than one filled at 20:00."""
    def key(t):
        aisle = int(layout.aisle_of_bay[t.bay])
        pos = float(layout.bay_xy[t.bay, 1])
        return aisle, pos if aisle % 2 == 0 else -pos
    dl = sorted([t for t in tasks_a if t.deadline is not None], key=key)
    high = sorted([t for t in tasks_a if t.deadline is None and t.tier == "high"], key=key)
    rest = sorted([t for t in tasks_a if t.deadline is None and t.tier != "high"], key=key)
    return dl + high + rest


def _timeline(seq: list[Task], a: Associate, layout: StoreLayout):
    out, loc, free = [], DEPOT, a.start
    for t in seq:
        tr = layout.travel_min(loc, t.bay)
        st = free + tr
        out.append(Assignment(t.task_id, a.id, st, st + t.duration, tr))
        loc, free = t.bay, st + t.duration
    return out


def _feasible(seq_sched: list[Assignment], seq: list[Task], a: Associate) -> bool:
    return all(s.end <= a.end and (t.deadline is None or s.end <= t.deadline) for s, t in zip(seq_sched, seq))


def cpsat(tasks: list[Task], staff: list[Associate], layout: StoreLayout, time_limit_s: float = 10.0,
          workers: int = 4, priority_scale: int = 10, travel_weight: int = 1, seed: int = 0, **_):
    """CP-SAT picks who does what (priority, duration, skills, shift and deadline capacity, aisle
    visits as the travel term); each associate's tasks are then walked in serpentine aisle order,
    with a repair step if exact walking times overflow a shift or deadline."""
    t0 = time.time()
    m = cp_model.CpModel()
    n = len(tasks)
    aisles = sorted({int(layout.aisle_of_bay[t.bay]) for t in tasks})
    lay_len = layout.aisle_len
    x, z = {}, {}
    travel_terms = []
    for a in staff:
        cap_min = a.end - a.start
        elig = [i for i, t in enumerate(tasks) if t.skill in a.skills and t.duration <= cap_min
                and (t.deadline is None or a.start + t.duration <= t.deadline)]
        for i in elig:
            x[a.id, i] = m.NewBoolVar(f"x_{a.id}_{i}")
        visit_cost = {}
        for g in aisles:
            z[a.id, g] = m.NewBoolVar(f"z_{a.id}_{g}")
            # walking into an aisle and back out to the cross aisle, plus a hop between aisles
            visit_cost[g] = math.ceil(lay_len / layout.walk_ft_per_min) + 1
        for i in elig:
            m.AddImplication(x[a.id, i], z[a.id, int(layout.aisle_of_bay[tasks[i].bay])])
        for g in aisles:
            members = [x[a.id, i] for i in elig if int(layout.aisle_of_bay[tasks[i].bay]) == g]
            if members:
                m.Add(z[a.id, g] <= sum(members))
            else:
                m.Add(z[a.id, g] == 0)
        travel = sum(visit_cost[g] * z[a.id, g] for g in aisles)
        travel_terms.append(travel)
        m.Add(sum(tasks[i].duration * x[a.id, i] for i in elig) + travel <= cap_min)
        dl_tasks = [i for i in elig if tasks[i].deadline is not None]
        for dl in sorted({tasks[i].deadline for i in dl_tasks}):
            grp = [i for i in dl_tasks if tasks[i].deadline <= dl]
            # deadline work is routed first; give it its own walking allowance
            dl_aisles = sorted({int(layout.aisle_of_bay[tasks[i].bay]) for i in grp})
            w_ = {g: m.NewBoolVar("") for g in dl_aisles}
            for i in grp:
                m.AddImplication(x[a.id, i], w_[int(layout.aisle_of_bay[tasks[i].bay])])
            m.Add(sum(tasks[i].duration * x[a.id, i] for i in grp)
                  + sum(visit_cost[g] * w_[g] for g in dl_aisles) <= dl - a.start)
    done = []
    for i, t in enumerate(tasks):
        mine = [x[k] for k in x if k[1] == i]
        d = m.NewBoolVar(f"done_{i}")
        m.Add(sum(mine) == d) if mine else m.Add(d == 0)
        done.append(d)
    m.Maximize(priority_scale * sum(t.weight * done[i] for i, t in enumerate(tasks))
               - travel_weight * sum(travel_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    status = solver.Solve(m)
    info = {"status": solver.StatusName(status), "solve_s": None,
            "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            "bound": solver.BestObjectiveBound(), "n_tasks": n, "repairs": 0, "fills": 0}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        info["solve_s"] = round(time.time() - t0, 3)
        return [], info
    unassigned = [t for i, t in enumerate(tasks) if not solver.Value(done[i])]
    routes = {}
    for a in staff:
        seq = _route([tasks[i] for (aid, i) in x if aid == a.id and solver.Value(x[aid, i])], a, layout)
        while seq and not _feasible(_timeline(seq, a, layout), seq, a):  # repair: drop least value per minute
            worst = min(seq, key=lambda t: (t.weight / t.duration, t.task_id))
            seq.remove(worst)
            unassigned.append(worst)
            info["repairs"] += 1
        routes[a.id] = seq
    info["fills"] = _fill(routes, unassigned, staff, layout)
    sched = [s for a in staff for s in _timeline(routes[a.id], a, layout)]
    info["solve_s"] = round(time.time() - t0, 3)
    return sched, info


def _fill(routes: dict, leftover: list[Task], staff: list[Associate], layout: StoreLayout) -> int:
    """Insert leftover tasks (best tier first) at the cheapest feasible position of any route."""
    rank = {tier: k for k, tier in enumerate(TIERS)}
    filled = 0
    for t in sorted(leftover, key=lambda t: (rank[t.tier], -t.weight / t.duration, t.task_id)):
        best = None
        for a in staff:
            if t.skill not in a.skills:
                continue
            seq = routes[a.id]
            base = sum(s.travel for s in _timeline(seq, a, layout))
            for pos in range(len(seq) + 1):
                cand = seq[:pos] + [t] + seq[pos:]
                tl = _timeline(cand, a, layout)
                if _feasible(tl, cand, a):
                    extra = sum(s.travel for s in tl) - base
                    if best is None or extra < best[0]:
                        best = (extra, a.id, cand)
        if best:
            routes[best[1]] = best[2]
            filled += 1
    return filled
