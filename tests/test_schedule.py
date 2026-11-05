import pytest

from storetask import schedule as scheduling
from storetask.layout import DEPOT
from storetask.schedule import Assignment, Associate
from storetask.tasks import Task


def make_task(task_id, bay, duration=10, tier="medium", skill="R", deadline=None):
    weights = {"high": 1000, "medium": 30, "low": 10}
    return Task(task_id, "restock", bay, duration, tier, weights[tier], skill, deadline, 1)


@pytest.fixture
def ample_instance(layout):
    staff = [
        Associate("A1", 0, 180, ("R", "P", "S")),
        Associate("A2", 0, 180, ("R", "P", "S")),
        Associate("A3", 20, 180, ("R", "P", "S")),
    ]
    tasks = [
        make_task("T0", 0, 8, "high", "P", 60),
        make_task("T1", 1, 9, "high", "P", 60),
        make_task("T2", 2, 10, "high", "R"),
        make_task("T3", 3, 11, "medium", "S"),
        make_task("T4", 4, 12, "medium", "R"),
        make_task("T5", 5, 13, "low", "P"),
        make_task("T6", 6, 14, "low", "S"),
        make_task("T7", 8, 15, "low", "R"),
    ]
    return tasks, staff, layout


@pytest.mark.parametrize("policy", ["fcfs", "greedy", "cpsat", "cpsat_circuit"])
def test_policies_return_feasible_complete_schedules_with_ample_capacity(policy, ample_instance):
    tasks, staff, layout = ample_instance
    if policy == "cpsat":
        schedule, _ = scheduling.cpsat(tasks, staff, layout, time_limit_s=2, workers=1)
    elif policy == "cpsat_circuit":
        schedule, _ = scheduling.cpsat_circuit(
            tasks, staff, layout, slot=1, time_limit_s=2, workers=1
        )
    else:
        schedule = getattr(scheduling, policy)(tasks, staff, layout)

    assert scheduling.validate(tasks, staff, layout, schedule) == []
    assert {row.task_id for row in schedule} == {task.task_id for task in tasks}


def _valid_assignment(task, associate, layout, start=None):
    travel = layout.travel_min(DEPOT, task.bay)
    start = associate.start + travel if start is None else start
    return Assignment(task.task_id, associate.id, start, start + task.duration, travel)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("travel", "starts"),
        ("overlap", "starts"),
        ("skill", "lacks skill"),
        ("shift", "ends after shift"),
        ("deadline", "misses deadline"),
        ("duplicate", "scheduled twice"),
        ("duration", "wrong duration"),
    ],
)
def test_validate_detects_each_violation_type(layout, case, message):
    associate = Associate("A", 0, 40, ("R",))
    task = make_task("T", 0, duration=10)
    tasks = [task]
    staff = [associate]
    valid = _valid_assignment(task, associate, layout)

    if case == "travel":
        schedule = [Assignment("T", "A", 0, 10, 0)]
    elif case == "overlap":
        other = make_task("U", 1, duration=10)
        tasks.append(other)
        schedule = [valid, Assignment("U", "A", valid.end - 1, valid.end + 9, 0)]
    elif case == "skill":
        tasks[0] = make_task("T", 0, duration=10, skill="P")
        schedule = [valid]
    elif case == "shift":
        tasks[0] = make_task("T", 0, duration=45)
        schedule = [_valid_assignment(tasks[0], associate, layout)]
    elif case == "deadline":
        tasks[0] = make_task("T", 0, duration=10, deadline=5)
        schedule = [_valid_assignment(tasks[0], associate, layout)]
    elif case == "duplicate":
        schedule = [valid, Assignment("T", "A", valid.end, valid.end + 10, 0)]
    else:
        schedule = [Assignment("T", "A", valid.start, valid.end + 1, valid.travel)]

    assert any(message in error for error in scheduling.validate(tasks, staff, layout, schedule))


def test_priority_policies_preserve_high_tier_work_when_overloaded(layout):
    staff = [Associate("A1", 0, 36, ("R",)), Associate("A2", 0, 36, ("R",))]
    tasks = [
        *[make_task(f"L{i}", 0, duration=15, tier="low") for i in range(4)],
        make_task("H0", 0, duration=10, tier="high"),
        make_task("H1", 0, duration=10, tier="high"),
    ]

    fcfs = scheduling.fcfs(tasks, staff, layout)
    greedy = scheduling.greedy(tasks, staff, layout)
    cpsat, _ = scheduling.cpsat(tasks, staff, layout, time_limit_s=2, workers=1)

    assert scheduling.summarize(tasks, staff, fcfs)["high_done"] < 2
    assert scheduling.summarize(tasks, staff, greedy)["high_done"] == 2
    assert scheduling.summarize(tasks, staff, cpsat)["high_done"] == 2
    assert scheduling.validate(tasks, staff, layout, fcfs) == []
    assert scheduling.validate(tasks, staff, layout, greedy) == []
    assert scheduling.validate(tasks, staff, layout, cpsat) == []


def test_summarize_counts_work_priority_and_capacity(layout):
    staff = [Associate("A", 0, 100, ("R",))]
    tasks = [
        make_task("H", 0, duration=10, tier="high"),
        make_task("M", 0, duration=20, tier="medium"),
        make_task("L", 0, duration=30, tier="low"),
    ]
    first = _valid_assignment(tasks[0], staff[0], layout)
    schedule = [first, Assignment("M", "A", first.end, first.end + 20, 0)]

    summary = scheduling.summarize(tasks, staff, schedule)

    assert summary == {
        "tasks": 3,
        "done": 2,
        "travel_min": first.travel,
        "work_min": 30,
        "demand_min": 60,
        "shift_min": 100,
        "weight_done": 1030,
        "weight_total": 1040,
        "high_total": 1,
        "high_done": 1,
        "medium_total": 1,
        "medium_done": 1,
        "low_total": 1,
        "low_done": 0,
    }
