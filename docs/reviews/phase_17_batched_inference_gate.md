# Phase 17 review: batched neural inference gate

## Why this precedes neural self-play

The scalar PUCT implementation is an intentionally simple correctness
reference.  In the mini neural arena, a 64x4 CNN completed only about five
simulations in a 50 ms move.  An RTX 3070 is poorly used by repeated
single-position calls; merely launching many independent Slurm jobs would
repeat that inefficiency and contend for GPUs.

The proposed production actor keeps multiple independent games in flight.
Each game selects one unevaluated leaf using the existing PUCT rules, then all
ready leaves are evaluated in one CNN batch.  Results return to their original
trees and use the existing absolute-value backup.  There is no virtual loss and
no concurrent traversal of one tree in the first implementation.

Before that coordinator is written, this gate measures the batch boundary
alone at sizes 1, 4, 8, 16, 32, and 64 on the actual HPC GPU and selected
checkpoint.  The result chooses a modest actor width from observed throughput
and latency rather than an arbitrary game count.

## Correctness boundary

`KerasEvaluator.evaluate()` now delegates to `evaluate_batch()` with one
state.  Both paths therefore share:

- mover-relative board encoding;
- the absolute-player identity plane;
- finite output and `[-1, 1]` value checks;
- legal-action masking; and
- stable softmax over legal logits only.

An HPC-only TensorFlow test compares batched and scalar policy/value outputs on
positions with different players to move.  The complete 76-test suite runs
before the timing script.

## Stop conditions and next step

This benchmark does not claim an end-to-end self-play speedup.  If batching
does not materially improve leaf throughput, the actor design must be
revisited before complex tree scheduling is added.  If it does, implement the
smallest lockstep coordinator and require exact scalar-versus-batched game
equivalence with a deterministic evaluator, including root noise, terminal
leaves, visit counts, and saved value statistics.

## HPC result and actor-width decision

Job 33521 passed all 78 tests on an RTX 3070. The selected standard 64x4
soft-Z network produced:

| Batch | Batch latency | Leaves/second |
| ---: | ---: | ---: |
| 1 | 11.25 ms | 88.9 |
| 4 | 11.44 ms | 349.8 |
| 8 | 11.64 ms | 687.0 |
| 16 | 12.06 ms | 1326.7 |
| 32 | 12.91 ms | 2479.2 |
| 64 | 14.63 ms | 4374.5 |
| 128 | 17.91 ms | 7148.8 |
| 256 | 24.80 ms | 10322.7 |

Batch 64 is about 49 times the batch-one leaf throughput for only 1.30 times
the call latency. We choose 64 active games for the first actor pilot. A
larger sweep is unnecessary at this gate: batch 64 already uses the GPU well,
fits naturally in one 64-game immutable chunk, and keeps the coordinator small.

The first end-to-end pilot uses 64 standard games, 16 simulations per move,
opening visit sampling, and no Dirichlet noise. Sixteen simulations is a
pipeline/throughput gate rather than a proposed training budget. Every raw
search statistic is still saved so the games remain auditable. Only after
checksum, reload, and scalar-invariant validation will we choose the 32/64
simulation production pilot.

The later color-bias diagnostic justified exact four-symmetry averaging. At
actor width 64 this becomes one batch of 256 transformed leaves, or 2581
effective leaves/second after dividing raw throughput by four. The ensemble
therefore preserves 59% of the non-ensemble actor rate while making policy and
absolute-value symmetries exact by construction.
