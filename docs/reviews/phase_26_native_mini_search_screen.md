# Phase 26 review: native mini gate and search screen

## What is authorized

Two cheap, independent checks may run after the audited branch is clean on the
cluster:

1. One RTX 3070 executes every test, including real native 5x5 and standard
   8x8 TensorFlow build/train/save/load boundaries.
2. Four CPU-only arena tasks screen tactical-rollout PUCT with `c_puct` values
   `0.25, 0.75, 1.5, 3.0` at identical 50 ms move budgets and the same 48
   paired openings. A 50 ms scheduling grace is outside the search budget and
   only distinguishes a genuinely late return from ordinary cluster jitter.

Parent-Q first-play urgency remains fixed. The screen changes no training data,
noise, temperature, architecture, or value target.

## Why this screen comes before more data

The pretraining labels come from MCTS. Generating hundreds of games with an
untested search constant would scale an assumption. The existing `1.5` result
is useful but was never compared with lower values; resource-efficient search
work explicitly warns that the useful scale can be setup-dependent.

This four-task run is a disaster screen, not a final tune. It compares each
PUCT variant to the same alpha-beta, random, and rollout anchors and records
paired uncertainty. If two settings are plausibly competitive, confirm those
two on fresh openings before choosing the 512-game data-generator setting.

## Failure checks

- Any nonterminal ply-limit result, timeout, illegal move, dirty worktree, or
  model/rules mismatch fails the phase.
- Ratings are not compared across tasks without inspecting their direct
  paired results against the immutable anchors.
- A point estimate with overlapping uncertainty does not select a winner.
- CPU arena tasks request no GPU; the TensorFlow gate requests exactly one.

## Execution note

Job 33538 passed all 88 tests, including the real native 5x5 and standard 8x8
TensorFlow boundaries, on an RTX 3070 in 18 seconds.

The first search attempt, array 33539, used a 20 ms move budget and 10 ms
grace. All four tasks correctly failed because 3--6 PUCT calls returned just
over the 30 ms adjudication threshold while four tasks shared one node. Those
games are retained as timing diagnostics but are inadmissible tuning evidence.
The revised 50/20 ms protocol must produce zero abnormal games. It uses fresh
openings and a new output directory; no point estimate from 33539 will select
a search constant.

Array 33543 then used the 50 ms search budget with 20 ms grace. The 0.25 and
1.5 tasks completed all 576 games normally. The 0.75 and 3.0 tasks each had
one return at 70.3 or 72.3 ms and correctly failed. Because `run_for_time()`
enforces the 50 ms internal deadline, these isolated overruns are consistent
with process descheduling, not extra search work. Retry only those two tasks,
at lower concurrency, on the identical opening seed with 50 ms adjudication
grace. All elapsed times remain in the records. A retry with any abnormal game
still fails; do not widen the grace again.

The retry array, 33547, completed both contaminated settings with zero
abnormal games. Combining only clean tasks gives the following direct
alpha-beta result (positive Elo is alpha-beta's advantage):

| `c_puct` | Alpha-beta W-L | Elo difference | 95% interval |
| ---: | ---: | ---: | ---: |
| 0.25 | 63-33 | +109.9 | [+9.1, +210.6] |
| 0.75 | 57-39 | +64.5 | [-33.1, +162.2] |
| 1.5 | 48-48 | 0.0 | [-96.1, +96.1] |
| 3.0 | 49-47 | +7.1 | [-89.0, +103.2] |

The screen rejects 0.25 and advances 1.5 and 3.0. It does not distinguish the
finalists; phase 28 compares them directly on fresh openings.
