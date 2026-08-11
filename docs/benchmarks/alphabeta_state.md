# Alpha-beta state-management benchmark

Date: 2026-08-11. Environment: local Windows Codex Python runtime. This is an
engineering routing measurement, not playing-strength evidence.

Command:

`python scripts/benchmark_alphabeta_state.py --positions 10 --repeats 5 --mini-depth 5 --standard-depth 4`

Each method searched identical ordered trees and returned identical values and
node counts. Method order alternated by round; the table reports median rates.

| Rules | Nodes/round | Clone child | Make/unmake | Make/unmake gain |
| --- | ---: | ---: | ---: | ---: |
| 5x5, depth 5 | 8,659 | 77,120 nodes/s | 92,696 nodes/s | 20.2% |
| 8x8, depth 4 | 91,938 | 31,038 nodes/s | 36,828 nodes/s | 18.7% |

Make/unmake was faster in every paired round. Absolute rates varied
substantially with local load: an earlier one-pass sequential timing even
suggested the opposite 8x8 conclusion. Alternating the order and comparing
paired rounds was therefore part of the correctness of the benchmark, not
presentation polish.

Decision: keep one mutable make/unmake path in alpha-beta. PUCT keeps lazy
cached child states because its branching ownership and measured result are
different. State-management strategy belongs to the search workload, not only
the game-state class.
