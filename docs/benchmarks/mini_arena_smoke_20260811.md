# Mini arena smoke check, 2026-08-11

This local run tested the evidence pipeline before its code was committed. It
used two saved 5x5 openings, four games per matchup, 5 ms per move, and all four
baseline variants. It is a smoke check, not a strength claim.

All 24 games ended by the rules: no timeout, illegal-move, agent-error, or ply-
limit result occurred. The slowest recorded move took 5.67 ms. Random lost all
12 games, matching the pre-registered prediction.

| First-agent view | W-L-D | Regularized Elo difference | Approx. 95% CI |
| --- | ---: | ---: | ---: |
| random vs alpha-beta | 0-4-0 | -381.7 | [-789.2, +25.8] |
| random vs PUCT rollout | 0-4-0 | -381.7 | [-789.2, +25.8] |
| random vs tactical PUCT | 0-4-0 | -381.7 | [-789.2, +25.8] |
| alpha-beta vs PUCT rollout | 4-0-0 | +381.7 | [-25.8, +789.2] |
| alpha-beta vs tactical PUCT | 2-2-0 | 0.0 | [-274.9, +274.9] |
| PUCT rollout vs tactical PUCT | 2-2-0 | 0.0 | [-274.9, +274.9] |

The regularization is one explicit virtual draw. Its purpose is to keep a
four-game sweep finite, not to make four games informative. The subsequent HPC
run uses 32 opening pairs (64 games per matchup) and saves the raw games.
