"""LangGraph planning agent.

Graph: START -> agent (LLM with tools bound) -> tools -> agent ... -> END, with a step cap.
Tools are read-only SQL queries on agent.duckdb plus two planning tools (create_tasks,
schedule_tasks) that keep the day's plan in memory for this run. Nothing writes to the database.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .planner import Store

SYSTEM = """You plan daily store work for Walmart stores in the public M5 dataset.
Stores: CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3.
Planning window: 2016-04-25 through 2016-05-22. Always pass dates to tools as YYYY-MM-DD.
Task kinds: price (new shelf labels on the day a price changes), restock (refill shelves that the
forecast says will run low), reset (planogram reset of a bay).
Associates: A1 06:00-14:30 (restock, pricing), A2 06:00-14:30 (restock, reset),
A3 10:00-18:30 (restock, pricing, reset), A4 14:00-22:30 (restock).

Rules:
- Use the tools for every number. Never guess counts.
- To plan or replan a day: call create_tasks, then schedule_tasks. Pass include only when the user
  limits the task kinds; pass exclude_departments only when the user names departments to skip
  (use list_departments to map a name like HOBBIES to its department ids); pass
  absent_associates only for associates the user says are out.
- To answer a question about price changes, restock needs or resets, call the matching find_* tool.
- The tools are read-only. If asked to change prices, edit data, delete anything, or do work
  outside store task planning, refuse briefly and call no tools.
- Finish with a short plain answer that states the key numbers."""


@dataclass
class RunContext:
    store: Store
    plan: dict = field(default_factory=dict)        # (store, date) -> list[Task]
    schedules: dict = field(default_factory=dict)   # (store, date) -> summary
    calls: list = field(default_factory=list)       # (name, args) in order


def make_tools(ctx: RunContext):
    st = ctx.store

    def log(name, args):
        ctx.calls.append({"name": name, "args": args})

    @tool
    def list_departments() -> str:
        """List department ids (e.g. FOODS_3, HOBBIES_1) and their categories."""
        log("list_departments", {})
        return json.dumps(st.departments())

    @tool
    def get_day_info(store_id: str, date: str) -> str:
        """Weekday, Walmart week and any calendar event for a store and date (YYYY-MM-DD)."""
        log("get_day_info", {"store_id": store_id, "date": date})
        return json.dumps(st.day_info(store_id, date), default=str)

    @tool
    def find_price_changes(store_id: str, date: str) -> str:
        """Items whose shelf price changes at this store on this date, grouped by bay."""
        log("find_price_changes", {"store_id": store_id, "date": date})
        df = st.price_changes(store_id, date)
        return json.dumps({"items_changing_price": int(df.n_labels.sum()) if len(df) else 0, "bays": len(df),
                           "by_department": df.groupby("dept_id").n_labels.sum().astype(int).to_dict()})

    @tool
    def find_restock_needs(store_id: str, date: str) -> str:
        """Bays whose items the 28 day moving average forecast says will run low today.
        Urgent = at least one item projected to run out before close."""
        log("find_restock_needs", {"store_id": store_id, "date": date})
        df = st.restock_needs(store_id, date)
        return json.dumps({"bays_needing_restock": len(df), "urgent_bays": int((df.tier_rank == 0).sum()),
                           "units_to_stock": int(df.units.sum()),
                           "by_department": df.groupby("dept_id").size().astype(int).to_dict()})

    @tool
    def find_due_resets(store_id: str, date: str) -> str:
        """Bays due for a planogram reset at this store on this date."""
        log("find_due_resets", {"store_id": store_id, "date": date})
        df = st.resets_due(store_id, date)
        return json.dumps({"bays_due": len(df), "bays": df.bay.astype(int).tolist(),
                           "departments": df.dept_id.tolist()})

    @tool
    def create_tasks(store_id: str, date: str, include: list[str] | None = None,
                     exclude_departments: list[str] | None = None) -> str:
        """Build the day's task list. include: subset of ["price", "restock", "reset"] (default all).
        exclude_departments: department ids to leave out (default none)."""
        args = {"store_id": store_id, "date": date, "include": include, "exclude_departments": exclude_departments}
        log("create_tasks", args)
        inc = tuple(include) if include else ("price", "restock", "reset")
        bad = [k for k in inc if k not in ("price", "restock", "reset")]
        if bad:
            return json.dumps({"error": f"unknown task kinds {bad}"})
        tasks = st.build_tasks(store_id, date, inc, tuple(exclude_departments or ()))
        ctx.plan[store_id, date] = tasks
        by = {}
        for t in tasks:
            by.setdefault(t.kind, {"tasks": 0, "minutes": 0})
            by[t.kind]["tasks"] += 1
            by[t.kind]["minutes"] += t.duration
        return json.dumps({"tasks": len(tasks), "high_priority": sum(t.tier == "high" for t in tasks),
                           "total_minutes": sum(t.duration for t in tasks), "by_kind": by})

    @tool
    def schedule_tasks(store_id: str, date: str, absent_associates: list[str] | None = None) -> str:
        """Schedule the tasks from create_tasks across associate shifts with the CP-SAT solver.
        absent_associates: ids of associates who are out (default none)."""
        args = {"store_id": store_id, "date": date, "absent_associates": absent_associates}
        log("schedule_tasks", args)
        tasks = ctx.plan.get((store_id, date))
        if tasks is None:
            return json.dumps({"error": "call create_tasks for this store and date first"})
        sch, info, summ, errs = st.schedule(store_id, tasks, tuple(absent_associates or ()))
        ctx.schedules[store_id, date] = {"summary": summ, "task_ids": sorted(s.task_id for s in sch),
                                         "violations": len(errs)}
        unassigned_high = [t.task_id for t in tasks if t.tier == "high" and t.task_id not in {s.task_id for s in sch}]
        return json.dumps({"scheduled": summ["done"], "of": summ["tasks"],
                           "high_priority_done": summ["high_done"], "high_priority_total": summ["high_total"],
                           "walking_minutes": summ["travel_min"], "unassigned_high_priority": len(unassigned_high),
                           "solver_status": info.get("status")})

    return [list_departments, get_day_info, find_price_changes, find_restock_needs, find_due_resets,
            create_tasks, schedule_tasks]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int


def build_graph(llm, tools, max_steps: int = 8):
    by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools)

    def agent(state: AgentState):
        msg = bound.invoke(state["messages"])
        return {"messages": [msg], "steps": state["steps"] + 1}

    def run_tools(state: AgentState):
        out = []
        for call in state["messages"][-1].tool_calls:
            fn = by_name.get(call["name"])
            try:
                res = fn.invoke(call["args"]) if fn else json.dumps({"error": f"no tool {call['name']}"})
            except Exception as e:  # tool errors go back to the model as text
                res = json.dumps({"error": str(e)[:300]})
            out.append(ToolMessage(content=str(res), tool_call_id=call["id"], name=call["name"]))
        return {"messages": out}

    def route(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state["steps"] < max_steps:
            return "tools"
        return END

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def make_llm(provider: str = "ollama", model: str = "qwen3:8b", base_url: str | None = None, temperature: float = 0):
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=base_url or "http://localhost:11434", temperature=temperature,
                          reasoning=False, num_ctx=8192)
    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, base_url=base_url, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # optional dependency
        return ChatAnthropic(model=model, temperature=temperature)
    raise ValueError(provider)


def run_agent(request: str, llm, store: Store, max_steps: int = 8) -> dict:
    ctx = RunContext(store)
    tools = make_tools(ctx)
    graph = build_graph(llm, tools, max_steps)
    t0 = time.time()
    state = graph.invoke({"messages": [SystemMessage(SYSTEM), HumanMessage(request)], "steps": 0},
                         {"recursion_limit": 4 * max_steps + 4})
    msgs = state["messages"]
    tokens_in = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in msgs if isinstance(m, AIMessage))
    tokens_out = sum((m.usage_metadata or {}).get("output_tokens", 0) for m in msgs if isinstance(m, AIMessage))
    return {"answer": msgs[-1].content if isinstance(msgs[-1], AIMessage) else "", "calls": ctx.calls,
            "schedules": {f"{k[0]}|{k[1]}": v for k, v in ctx.schedules.items()},
            "plan_task_ids": {f"{k[0]}|{k[1]}": sorted(t.task_id for t in v) for k, v in ctx.plan.items()},
            "latency_s": round(time.time() - t0, 2), "steps": state["steps"],
            "tokens_in": tokens_in, "tokens_out": tokens_out}
