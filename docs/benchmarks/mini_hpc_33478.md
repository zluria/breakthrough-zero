# Mini baseline tournament: HPC job 33478

This was the first pre-registered 5x5 baseline tournament. It is preserved as
evidence, including the opening-quality problem it revealed. Raw games and the
original generated summary are in [`../../results/mini-hpc-33478/`](../../results/mini-hpc-33478/).

## Reproducibility and resources

- Commit: `4e957d518a86ba5039b45d049380e26872d4fd3b`
- Slurm job: `33478`, exit `0:0`, elapsed 73 seconds
- Host: `HPC-RTX3070-08`
- Allocation: CPU-only, 2 GB requested, 42 MB peak RSS, no GPU TRES
- Rules: 5x5, one starting row
- Openings: 32 distinct six-ply prefixes, eight dummy-PUCT simulations
- Match: 32 color-reversed pairs per matchup, 50 ms per rated move
- Scope: one opening seed and one tournament run

Job `33477` was the preserved failed attempt. It played zero games: the test
gate selected system Python 3.9 and stopped on missing NumPy in zero scheduler
seconds. Job 33478 used the validated Python 3.11 executable directly.

## Raw audit

All 384 games replay to the recorded terminal winner. There were no draws,
forfeits, illegal moves, agent exceptions, or ply-limit adjudications. Every
opening pair reverses colors and reuses the corresponding agent seeds. PUCT
usually finished its current complete simulation just after 50 ms; the worst
move was 62.84 ms, below the predeclared 70 ms forfeit threshold.

The table below recomputes uncertainty using opening pairs as the independent
units. It supersedes only the original summary's narrower game-level interval;
raw wins, losses, and scores are unchanged.

| First-agent view | W-L-D | Pair-regularized Elo | 95% CI |
| --- | ---: | ---: | ---: |
| random vs alpha-beta | 1-63-0 | -602.1 | [-907.1, -297.0] |
| random vs PUCT rollout | 2-62-0 | -528.9 | [-788.6, -269.2] |
| random vs tactical PUCT | 1-63-0 | -602.1 | [-907.1, -297.0] |
| alpha-beta vs PUCT rollout | 46-18-0 | +157.3 | [+29.3, +285.3] |
| alpha-beta vs tactical PUCT | 34-30-0 | +21.1 | [-95.5, +137.6] |
| PUCT rollout vs tactical PUCT | 19-45-0 | -144.7 | [-270.9, -18.5] |

## Diagnostic finding

Random's four wins were legal, but each paired game was won by the same
absolute color. One random win happened on its first rated move because the
six-ply noisy prefix already offered Player 1 an immediate goal move. A fixed
opening length does not transfer safely to a shorter game: six plies consume a
large fraction of 5x5 Breakthrough's useful horizon.

The completed diagnostic rerun is in
[`mini_hpc_33479.md`](mini_hpc_33479.md). It uses four uniform-random plies,
exactly two moves per side, and no Dirichlet search noise. The first run is not
deleted or silently replaced.
