# Store Task Planning Agent and Associate Shift Scheduling (Walmart M5)

Turns public Walmart M5 item demand and real week over week shelf price changes into a store's daily
restock, price label and planogram reset tasks, schedules those tasks across associate shifts with an
OR-Tools CP-SAT model, and wraps both behind a LangGraph agent that plans a day from a plain request
("A2 called out at CA_3 on May 10, skip HOBBIES") by calling read-only SQL tools on DuckDB.

**Live summary:** https://abhayjuloori.github.io/store-task-agent-cpsat-m5/

## Results at a glance
All numbers from `docs/data/run_report.json` (full run, 2,517 s, 6/6 checks) and `eval/results/`.

| | 4 associates | 3 associates (A4 out) |
|---|---|---|
| Task minutes / shift minutes | 0.81 | 1.31 |
| High priority tasks done: FCFS / greedy / **CP-SAT** (paired, 280 store days) | 99.1% / 99.97% / **99.97%** | 77.3% / 87.2% / **92.4%** |
| CP-SAT minus FCFS, high priority coverage (bootstrap 95% CI) | +0.7 pts [0.4, 1.0] | **+14.7 pts [13.9, 15.3]** |
| CP-SAT minus greedy | +0.0 pts [-0.0, 0.0] | +4.6 pts [4.0, 5.3] |
| Demand lost to empty shelves, 28 day closed loop: FCFS / greedy / CP-SAT | 4.93% / **4.52%** / 4.92% | 14.89% / **7.25%** / 9.83% |
| Walking minutes per completed task: FCFS / greedy / CP-SAT | 1.04 / 1.03 / 1.11 | 1.04 / 1.01 / 1.06 |

- **CP-SAT finishes the most high priority work when a store is short staffed**, 14.7 points more than
  first come first served on identical task lists, and beats the priority greedy rule on 144 of 280
  short staffed store days (loses on 20).
- **But the greedy rule loses the fewest units on real M5 sales.** The CP-SAT objective counts high
  priority *tasks*, not *units at risk*, so it packs many short tasks and leaves some long, high volume
  restocks; it also starves resets (reset backlog 489 bays at day 28 vs 173 for greedy). This is the
  clearest next step: weight restocks by expected units saved and add a time of day term.
- **No walking win.** CP-SAT walks slightly more per task than FCFS. Walking is about one minute per
  task in this layout, so priority dominates the objective.
- **Solver:** median 3.06 s per store day (3 s limit), 46% proven optimal, 131 tasks median (162 max).
- **Agent (Kaggle T4, Ollama):** on the 19 request holdout, v1 17/19 for both models, **v2 19/19** for
  both. On the 60 request main set, qwen3:14b went from 90.0% to 96.7% and qwen3:8b from 76.7% to 90.0%.


## Problem
A store manager starts the day with three kinds of work: shelves the forecast says will run low,
shelf labels that must show a new price by noon on the day it changes, and bays due for a planogram
reset. There are more task minutes than associate minutes on busy days, associates have different
skills and shift windows, and walking between aisles is not free. The questions:

1. Which tasks does the day actually need, computed from data rather than guessed?
2. Given the staff on shift, which tasks get done, by whom, in what order?
3. How much better is an optimizer than the rule a store would otherwise use (first come first served)?
4. Can an LLM agent turn a manager's request into the right tool calls and arguments?

## Data
| Source | Real or synthetic | Used for |
|---|---|---|
| M5 `sales_train_evaluation.csv` (10 stores, 3,049 items, daily units) | real | 28 day moving average forecast; actual units sold in the shelf simulation |
| M5 `sell_prices.csv` (weekly price per store and item) | real | price label tasks: every item whose price differs from the previous Walmart week (854 changes in the 28 day window across 10 stores) |
| M5 `calendar.csv` | real | dates, Walmart weeks (start Saturday), events |
| Store layout, shelf capacity, task durations, shifts, skills, reset calendar | **synthetic**, seeded, all in `configs/assumptions.yaml` | everything M5 does not contain |

Planning window: the M5 evaluation period, 2016-04-25 to 2016-05-22 (28 days), all 10 stores.
See `data/README.md` for checksums and the download mirror.

## Method

### 1. DuckDB load and task generation (`load.py`, `tasks.py`, `agent/planner.py`)
- Sales unpivoted to long form from 2015-03-08 (13,476,580 rows); price changes found with a `lag()`
  window over weekly prices (78,815 changes in the full history). Load takes about 9 s.
- **Layout (synthetic):** each store's 7 departments are placed in a seeded order; items fill bays of
  24 in item id order (128 bays, 16 aisles of 8 bays). Walking minutes come from aisle geometry: same
  aisle = distance along it, different aisles = around the shorter end, at 150 ft/min with a cart.
- **Restock:** shelf capacity = 4 days of the item's mean demand over the 90 days before the window.
  Each morning the forecast is the mean of the previous 28 days of actual units. An item is *high*
  priority if its projected end of day level is below zero, *medium* if it falls below 35% of
  capacity. Items are grouped into one task per bay; duration = 6 min setup + 0.12 min/unit + 0.4 min/item.
- **Price labels:** every real price change, one task per bay, 3 min + 1.5 min/label, high priority,
  must finish by 12:00 on the effective day; an unfinished one carries to the next day.
- **Resets:** 2 bays a day from a seeded store order, 45 min each, medium priority, reset skill only.
- The same rules are written twice: as SQL (`agent/planner.py`, what the agent's tools run) and in
  numpy (`sim.py`, for speed). The run checks they produce identical task sets for every store and day.

### 2. Scheduling (`schedule.py`)
Four synthetic shifts: A1 and A2 06:00-14:30, A3 10:00-18:30, A4 14:00-22:30, with restock / pricing /
reset skills. Three policies:

| Policy | Rule |
|---|---|
| FCFS | tasks in the order the system emits them (price labels, restock by bay, resets); each goes to the skilled associate free soonest; skipped if it no longer fits the shift or deadline |
| Greedy | when an associate frees up, they take the highest tier task they can finish, nearest first |
| CP-SAT | assignment model: one boolean per (associate, task); shift capacity and noon deadline capacity per associate; aisle visit booleans carry the walking cost; objective = 10 x priority weight done - walking minutes (high = 1000, medium = 30, low = 10, so one high task outweighs any realistic number of lower ones). Each associate's tasks are then walked deadline first, high tier next, rest in serpentine aisle order; a repair step drops the least valuable minute if exact walking overflows a shift, and a fill step inserts leftovers wherever they fit. 3 s limit, 4 workers |

Every schedule from every policy is re-checked by an independent validator (skills, shift windows,
deadlines, walking time between consecutive tasks, no double booking).

**Why not one big routing model?** The first version was an exact vehicle routing model (a CP-SAT
circuit per associate, minute level start times, walking on every arc). At about 130 tasks and 4
associates it has about 70,000 arc literals; in 10 s it barely improved on the greedy warm start, and
putting start times on a 15 minute grid (the usual trick to make it smaller) made it worse still,
because most tasks take 8 to 20 minutes and every short task leaves idle time before the next slot.
The ablation table below measures this on one day per store.

### 3. Evaluation design (`sim.py`, `run.py`)
- **Paired (open loop):** for every store and day, all three policies schedule the *identical* task
  list; shelves advance with the FCFS schedule (the status quo), so the task list does not depend on
  the policy being scored. 10 stores x 28 days x 2 staffing levels.
- **Closed loop:** each policy runs its own 28 day simulation. Completed restocks fill shelves to
  capacity at the task's finish minute; real M5 units then sell evenly over store hours (06:00-23:00);
  demand that meets an empty shelf counts as lost units. This is where scheduling choices show up as
  an outcome rather than an objective value.
- **Staffing:** 4 associates (base) and 3 (A4 out), both fixed before any policy was compared.

### 4. Planning agent (`agent/graph.py`)
A LangGraph `StateGraph`: `agent` node (chat model with tools bound) -> `tools` node -> back to
`agent`, stopping when the model answers or after 8 steps. Seven tools, all read-only SQL on a 15 MB
DuckDB file (`agent.duckdb`: calendar, planogram, recent sales, price changes, morning shelf
snapshots from the status-quo simulation, reset calendar, staff): `list_departments`, `get_day_info`,
`find_price_changes`, `find_restock_needs`, `find_due_resets`, `create_tasks(include,
exclude_departments)`, `schedule_tasks(absent_associates)`. The plan lives in memory for the run;
nothing writes to the database. Model: `qwen3:8b` (and `qwen3:14b` for comparison) served by Ollama;
`make_llm` also takes any OpenAI compatible endpoint or the Anthropic API.

**Scenario set** (`eval/scenarios.jsonl`, 60 requests, generated with seed 7 before any agent run was
scored): 12 full day plans, 6 price-only, 8 skip-a-department, 8 associate call-outs, 6 combined
constraints, 13 count questions, 7 requests the agent must refuse (change a price, drop a table,
email someone, off-topic). Dates appear as ISO, "May 7, 2016" and "Saturday, May 7 2016".
Scoring: tool selection, exact arguments after normalising defaults, final task set equal to the
gold run, every gold number present in question answers, and zero tool calls on refusals.

## Detailed results

### Model ablation (one overloaded day per store, 3 associates, 10 s per solve)
| Variant | High priority done (of 943) | All tasks done | Walking min |
|---|---|---|---|
| Priority greedy | 603 | 618 | 628 |
| **CP-SAT assignment + route (main)** | **646** | 671 | 755 |
| CP-SAT circuit, minute start times | 510 | 627 | 643 |
| CP-SAT circuit, minute starts, greedy warm start | 605 | 622 | 624 |
| CP-SAT circuit, 15 minute start grid | 442 | 515 | 539 |

The time limit was set to 3 s after one overloaded day (CA_3, 2016-04-30, 3 associates) gave 62, 62,
62, 61 and 61 high priority tasks at 1, 3, 5, 10 and 30 s.

### Planning agent
| Run | Requests | Success | Tool selection | Planning args | Question answers | Refusals | Median latency |
|---|---|---|---|---|---|---|---|
| v1 qwen3:8b, main | 60 | 76.7% | 83.3% | 65.0% | 100% | 100% | 5.3 s |
| v1 qwen3:14b, main | 60 | 90.0% | 98.3% | 87.5% | 100% | 85.7% | 16.5 s |
| v1 qwen3:8b, holdout | 19 | 89.5% | 100% | 92.3% | 100% | 100% | 7.5 s |
| v1 qwen3:14b, holdout | 19 | 89.5% | 100% | 84.6% | 100% | 100% | 19.3 s |
| **v2 qwen3:8b, holdout** | 19 | **100%** | 100% | 100% | 100% | 100% | 6.6 s |
| **v2 qwen3:14b, holdout** | 19 | **100%** | 100% | 100% | 100% | 100% | 17.2 s |
| v2 qwen3:8b, main (optimistic) | 60 | 90.0% | 100% | 85.0% | 100% | 100% | 6.8 s |
| v2 qwen3:14b, main (optimistic) | 60 | 96.7% | 98.3% | 97.5% | 100% | 85.7% | 15.3 s |

What changed from v1 to v2, all from the v1 failure review:
- **qwen3:8b returned an empty reply on all 8 call-out requests.** Not the model: with
  `langchain-ollama` 1.1.0 the tool call was dropped, while Ollama's streaming API and its OpenAI
  compatible `/v1` endpoint both returned a correct `schedule_tasks` call. v2 talks to Ollama through
  `/v1` and retries an empty turn once.
- **"Skip the FOODS department" excluded nothing.** The model passed the category name and the v1 tool
  treated unknown ids as a no-op. v2 resolves category names to department ids and returns an error
  for anything unknown; associate ids are validated the same way.
- **qwen3:14b planned a day when asked to email a schedule.** The prompt now says the tools are read
  only and cannot send messages. (It still did on the main set: refusals 6/7.)

The holdout (`eval/scenarios_holdout.jsonl`, new phrasings: lowercase store ids, vacations, roped off
aisles, two departments at once, four new refusals) was written after the v1 review and before any
v2 run, so v1 vs v2 on it is the fair comparison. The v2 main-set numbers are optimistic because the
fixes were shaped on main-set failures. v2 also introduced a new failure: on 5 main-set requests
qwen3:8b passed all four associates as absent, so nothing was scheduled.

## Limitations
- Store layout, shelf capacity, task durations, shifts, skills and the reset calendar are synthetic.
  Demand, prices and dates are real M5 data.
- The CP-SAT objective (priority count) and the outcome (lost units) disagree; see Results.
- The restock forecast is a 28 day moving average, chosen for transparency, not accuracy.
- Shelf simulation: demand spread evenly over store hours, back room stock unlimited, restock fills
  to capacity, no shrink or phantom inventory.
- The assignment model approximates walking with aisle visits; exact walking is applied afterwards by
  the route, repair and fill steps.
- Agent requests are generated from templates plus a 19 request holdout, not real manager messages.
  Gold answers are exact because the tools are deterministic, but the request distribution is narrow.
- `make_llm` supports the Anthropic API, but every reported run used qwen3 through Ollama.
- M5 files come from a public Kaggle mirror (sizes match the competition files; checksums in
  `data/README.md`).


## How to run
```bash
uv venv -p 3.11 .venv && uv pip install -p .venv -e ".[dev]"
scripts/download_m5.sh                                  # Kaggle CLI, about 450 MB
PYTHONPATH=src .venv/bin/python -m storetask.run          # load + all simulations + ablation (~40 min on 4 cores)
PYTHONPATH=src .venv/bin/python -m storetask.run --quick  # 2 stores x 7 days
PYTHONPATH=src .venv/bin/python -m storetask.agent.data   # build agent.duckdb
ollama pull qwen3:8b
PYTHONPATH=src .venv/bin/python -m storetask.agent.evaluate --model qwen3:8b
PYTHONPATH=src .venv/bin/python -m pytest -q
```
The agent evaluation in the results table ran on a Kaggle 2x T4 kernel (`scripts/kaggle/`), one
Ollama server per GPU.

## Repository
```
configs/assumptions.yaml   every synthetic number
src/storetask/             load, layout, tasks, schedule, sim, run
src/storetask/agent/       agent.duckdb builder, SQL planner, LangGraph graph, scenarios, scorer
eval/                      scenarios and agent results
docs/                      GitHub Pages summary and run_report.json
tests/                     pytest suite
```
