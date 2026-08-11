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
