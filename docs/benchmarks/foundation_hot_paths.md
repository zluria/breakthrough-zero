# Foundation hot-path benchmark

Date: 2026-08-11. Environment: local Windows Codex Python runtime. These are
engineering routing measurements, not playing-strength claims. HPC profiling
must repeat them before a large run.

## Rules and rollout sampling

Command:

`python scripts/benchmark_rules.py --positions 200 --repeats 50 --games 300`

| Operation | Rate |
| --- | ---: |
| Literal reference legal moves | 44,641 calls/s |
| Bitboard legal moves | 67,910 calls/s |
| Build legal list then choose | 62,602 choices/s |
| Bit-sample one uniform move | 415,607 choices/s |
| Bit-sample with win/capture preference | 214,938 choices/s |
| List-based complete rollouts | 847 games/s |
| Bit-sampled uniform rollouts | 2,967 games/s |
| Tactical bit-sampled rollouts | 1,896 games/s |

The bit sampler removes avoidable `Move` allocations and is the uniform
baseline. The tactical sampler remains optional: it is faster than the original
list path but slower than optimized uniform sampling and produced longer games
(75.1 versus 63.3 plies in this small run).

## State transitions

Command:

`python scripts/benchmark_state_management.py --repeats 5000 --depths 4 8 16 32`

| Depth | Clone + replay | Reset + replay | Make + unmake |
| ---: | ---: | ---: | ---: |
| 4 | 10.0 us | 10.0 us | 11.7 us |
| 8 | 19.5 us | 19.1 us | 21.6 us |
| 16 | 40.0 us | 38.5 us | 44.6 us |
| 32 | 77.3 us | 78.0 us | 87.6 us |

A lazy child clone-plus-move cost roughly 2.7--3.4 microseconds once per visited
node. Conservative cached-state size was about 220 bytes.

## End-to-end scalar PUCT state strategy

The first isolated run was noisy enough to suggest the wrong conclusion, so the
final comparison alternated both methods in one process and used median rates.

Command:

`python scripts/benchmark_search.py --total-simulations 5000 --budgets 32 100 400 --rounds 3`

| Simulations/search | Replay sims/s | Lazy cache sims/s | Faster method |
| ---: | ---: | ---: | --- |
| 32 | 10,753 | 9,768 | Replay by 10.1% |
| 100 | 6,917 | 7,435 | Cache by 7.5% |
| 400 | 6,995 | 7,562 | Cache by 8.1% |

Decision: use lazy cached states for intended full searches. Do not add a second
fast-search state path yet. Reconsider it only when playout-cap randomization is
implemented and an end-to-end profile shows the roughly 10% low-budget gain is
material relative to evaluator cost and code complexity.
