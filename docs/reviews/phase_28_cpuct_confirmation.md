# Phase 28 review: direct cPUCT finalist confirmation

## Why another match is needed

The four-way screen runs each exploration constant in a separate tournament
against common fixed anchors. It can reject poor settings, but overlapping
anchor intervals do not establish an ordering between close settings. Fitted
global Elo values from separate tournaments are not a substitute for games
between the candidates.

After every screen task is free of abnormal games, select the two plausible
best tactical-rollout PUCT settings by their direct results against alpha-beta
and the plain-rollout anchor. Play those finalists against each other on 96
new four-ply opening pairs, once with each color. Both receive the same 50 ms
internal search budget; 50 ms external grace handles cluster descheduling but
adds no search work. Search noise is disabled. Agent and opening seeds, every
move time, and every completed simulation count are retained.

## Decision rule

- Any abnormal game invalidates the run.
- A clear paired result promotes its winner to the 512-game baseline corpus.
- If the 95% interval overlaps zero, use the simpler status quo `c_puct=1.5`
  rather than repeatedly sampling openings until a preferred candidate wins.
- The promoted constant is a 5x5 starting point, not an 8x8 truth and not a
  substitute for later simulation-budget and neural-search checks.

This confirmation changes only `c_puct`. FPU remains parent Q, the evaluator
and clock are identical, and no architecture, target, sampling, or noise
choice is bundled into the comparison.

The clean anchor screen selected 1.5 and 3.0 as the two finalists. Alpha-beta
scored 48-48 and 49-47 against them respectively, with both 95% intervals
spanning zero. No other screened setting was comparably strong against that
anchor.

## Result

Job 33549 completed all 192 games normally. `c_puct=1.5` scored 90-102
against 3.0, or -21.5 Elo with a 95% interval of [-90.3, +47.3]. Pair results
were 12 sweeps for 1.5, 18 for 3.0, and 66 color splits.

The interval overlaps zero, so the preregistered rule retains the status quo
`c_puct=1.5`. The experiment has answered its bounded question; do not buy
more games to force a distinction. This is the explicit search constant for
the phase-27 corpus and remains a 5x5 starting point rather than a universal
default.
