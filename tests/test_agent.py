import json
from collections.abc import Sequence

import pandas as pd
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from storetask.agent.evaluate import score, summarize
from storetask.agent.graph import RunContext, build_graph, make_tools
from storetask.agent.scenarios import build as build_scenarios
from storetask.schedule import Assignment
from storetask.tasks import Task


class ScriptedChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Sequence, **kwargs):
        return self


class FakeStore:
    def __init__(self):
        self.task = Task("S-d1-R-0", "restock", 0, 10, "high", 1000, "R", None, 1)

    def departments(self):
        return [{"dept_id": "D1", "cat_id": "D"}]

    def day_info(self, store_id, date):
        return {"store_id": store_id, "date": date, "weekday": "Sunday"}

    def price_changes(self, store_id, date):
        return pd.DataFrame(
            {"n_labels": [2, 3], "dept_id": ["D1", "D1"], "bay": [0, 1]}
        )

    def restock_needs(self, store_id, date):
        return pd.DataFrame(
            {"tier_rank": [0, 1], "units": [4, 2], "dept_id": ["D1", "D1"]}
        )

    def resets_due(self, store_id, date):
        return pd.DataFrame({"bay": [1, 4], "dept_id": ["D1", "D1"]})

    def build_tasks(self, store_id, date, include, exclude_departments):
        return [self.task]

    def schedule(self, store_id, tasks, absent=()):
        schedule = [Assignment(tasks[0].task_id, "A", 1, 11, 1)]
        summary = {
            "tasks": 1,
            "done": 1,
            "high_done": 1,
            "high_total": 1,
            "travel_min": 1,
        }
        return schedule, {"status": "OPTIMAL"}, summary, []


def ai_tool(name, args, call_id):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def invoke_script(messages, tools, max_steps=8):
    model = ScriptedChatModel(messages=iter(messages))
    graph = build_graph(model, tools, max_steps=max_steps)
    return graph.invoke({"messages": [HumanMessage("plan")], "steps": 0})


def test_graph_routes_create_then_schedule_then_final_answer():
    ctx = RunContext(FakeStore())
    tools = make_tools(ctx)
    args = {"store_id": "S", "date": "2016-05-01"}

    state = invoke_script(
        [
            ai_tool("create_tasks", args, "create"),
            ai_tool("schedule_tasks", args, "schedule"),
            AIMessage(content="Scheduled one high-priority task."),
        ],
        tools,
    )

    assert [call["name"] for call in ctx.calls] == ["create_tasks", "schedule_tasks"]
    assert state["steps"] == 3
    assert state["messages"][-1].content == "Scheduled one high-priority task."


def test_graph_honors_max_steps_before_running_more_tools():
    ctx = RunContext(FakeStore())
    tools = make_tools(ctx)
    args = {"store_id": "S", "date": "2016-05-01"}

    state = invoke_script(
        [
            ai_tool("create_tasks", args, "create"),
            ai_tool("schedule_tasks", args, "schedule"),
        ],
        tools,
        max_steps=2,
    )

    assert state["steps"] == 2
    assert [call["name"] for call in ctx.calls] == ["create_tasks"]


def test_tool_exceptions_are_returned_as_tool_message_text():
    @tool
    def explode() -> str:
        """Raise a test error."""
        raise ValueError("boom")

    state = invoke_script(
        [ai_tool("explode", {}, "explode"), AIMessage(content="handled")],
        [explode],
    )
    messages = [message for message in state["messages"] if isinstance(message, ToolMessage)]

    assert len(messages) == 1
    assert json.loads(messages[0].content) == {"error": "boom"}


def test_schedule_before_create_returns_error_tool_message():
    ctx = RunContext(FakeStore())
    tools = make_tools(ctx)
    args = {"store_id": "S", "date": "2016-05-01"}

    state = invoke_script(
        [ai_tool("schedule_tasks", args, "schedule"), AIMessage(content="fixed")],
        tools,
    )
    message = next(message for message in state["messages"] if isinstance(message, ToolMessage))

    assert ctx.calls[0]["name"] == "schedule_tasks"
    assert json.loads(message.content) == {"error": "call create_tasks for this store and date first"}


def base_run(**overrides):
    run = {
        "answer": "",
        "calls": [],
        "schedules": {},
        "plan_task_ids": {},
        "latency_s": 0.1,
        "steps": 1,
        "tokens_in": 10,
        "tokens_out": 5,
    }
    run.update(overrides)
    return run


def test_evaluate_score_and_summary_for_planning_question_and_refusal():
    store = FakeStore()
    planning = {
        "id": "full_plan-01",
        "kind": "full_plan",
        "gold": {
            "store": "S",
            "date": "2016-05-01",
            "include": ["reset", "price", "restock"],
            "exclude": [],
            "absent": [],
        },
    }
    key = "S|2016-05-01"
    planning_run = base_run(
        answer="Done",
        calls=[
            {
                "name": "create_tasks",
                "args": {
                    "store_id": "S",
                    "date": "2016-05-01",
                    "include": None,
                    "exclude_departments": None,
                },
            },
            {
                "name": "schedule_tasks",
                "args": {"store_id": "S", "date": "2016-05-01", "absent_associates": None},
            },
        ],
        schedules={key: {"violations": 0}},
        plan_task_ids={key: [store.task.task_id]},
    )
    planning_score = score(planning, planning_run, store)
    assert planning_score["args_ok"]
    assert planning_score["success"]

    question = {
        "id": "q_price-01",
        "kind": "q_price",
        "gold": {"store": "S", "date": "2016-05-02", "tool": "find_price_changes"},
    }
    question_score = score(
        question,
        base_run(
            answer="There are 5 price labels to change.",
            calls=[
                {
                    "name": "find_price_changes",
                    "args": {"store_id": "S", "date": "2016-05-02"},
                }
            ],
        ),
        store,
    )
    assert question_score["answer_ok"]
    assert question_score["success"]

    refusal = {"id": "refuse-01", "kind": "refuse", "gold": {}}
    refusal_score = score(refusal, base_run(answer="I cannot do that."), store)
    assert refusal_score["success"]

    summary = summarize([planning_score, question_score, refusal_score])
    assert summary["n"] == 3
    assert summary["success"] == 1.0
    assert summary["tool_selection"] == 1.0
    assert summary["planning_args"] == 1.0
    assert summary["planning_plan_match"] == 1.0
    assert summary["question_answer"] == 1.0
    assert summary["refusal"] == 1.0
    assert summary["schedule_violations"] == 0


def test_scenarios_are_deterministic_unique_and_in_window():
    rows = build_scenarios()
    assert rows == build_scenarios()
    assert len(rows) == 60
    assert len({row["id"] for row in rows}) == 60

    for row in rows:
        date = row["gold"].get("date")
        if date is not None:
            assert "2016-04-25" <= date <= "2016-05-22"
        if row["kind"] in {"q_price", "price_only"}:
            assert pd.Timestamp(date).day_name() == "Saturday"
