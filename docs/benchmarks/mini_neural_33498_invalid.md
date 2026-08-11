# Invalid neural screening tournament: job 33498

Job `33498` completed 896 mini-board games in 5 minutes 32 seconds on one
RTX3070.  All model loading, tests, and file writes passed, but the tournament
is **invalid as playing-strength evidence**.

## Failure found by trajectory audit

The raw match files contain 748 ordinary terminal games and 148
`time_forfeit` games:

| Agent blamed by final record | Time forfeits |
| --- | ---: |
| Tactical rollout PUCT | 85 |
| Plain rollout PUCT | 59 |
| Tiny outcome model | 2 |
| Base outcome model | 1 |
| Base soft-Z model | 1 |

The rated move budget was 50 ms with 30 ms tolerance.  Rollout PUCT averaged
about 55 ms per move, but late moves averaged roughly 100--106 ms.  Neural
agents averaged 7--26 simulations per move; their four forfeits show that
runtime jitter also existed outside rollouts.

The search timer previously began another complete simulation whenever any
positive time remained.  A long rollout or scheduler delay could therefore
cross the arena deadline.  TensorFlow thread pools and three concurrent CPU
probe tasks made the tail easier to expose.  The Elo table in the raw output is
retained only so the failure is reproducible.

## Corrective actions

1. Timed PUCT now measures the previous complete simulation and stops when
   1.25 times that cost is unlikely to fit in the remaining budget.
2. Tournament summaries count every termination class and the command fails by
   default after saving results if any abnormal game occurs.
3. The neural Slurm job caps TensorFlow and OpenMP CPU thread pools.
4. A local 48-game gate then completed with zero abnormal terminations under a
   tighter 20 ms tolerance.

The fixed-data training curves and saved models are unaffected.  Only job
`33498`'s game outcomes and ratings are rejected.  Its complete raw files are
kept in [`results/mini-neural-33498`](../../results/mini-neural-33498).
