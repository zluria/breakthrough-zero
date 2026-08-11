# Project status and experiment ledger

Updated 2026-08-11. This is the control surface; phase reviews contain the
detail. “Validated” means the stated evidence exists, not that a method has
been established as strongest.

## Current gate

| Item | Status | Evidence / next action |
| --- | --- | --- |
| Rules, terminal semantics, absolute P1 values | Validated | Reference-generator, terminal, symmetry, and hand-tree tests |
| Dummy evaluator and PUCT sanity check | Validated | Seeded absolute rollouts; immediate-win/capture preference; parent-Q FPU tests |
| Hybrid node state storage | Validated | Clone-on-first-visit beat replay and make/unmake in the recorded microbenchmark |
| Native 5x5 input and 75-action head | Validated | Job 33538 passed all 88 tests plus native generate/train/save/load on RTX 3070 |
| 8x8 input and 192-action compatibility | Validated | Job 33538 passed real TensorFlow shape and cross-rules rejection tests |
| Rules-derived game bound | Validated locally | Potential proof gives 40 mini and 208 standard plies |
| Four training symmetries | Validated locally | Balanced four-epoch cycle; identity-only validation |
| Outcome / soft-Z / 50:50 mixture | Implemented locally | Fixed-data 5x5 comparison is next; no winner declared |
| Literature/code survey | Complete for this gate | OLIVAW, KataGo, Gumbel, Mctx/Pgx, Leela, cross-size GNN, search control, value targets |
| Independent audit | Received | [External report](reviews/external_audit_20260811.md); another fresh review is required before expensive 8x8 scaling |

## Results whose scope changed

| Artifact | Status now | Reason |
| --- | --- | --- |
| Job 33516, 896-game mini neural screen | Raw data valid; neural conclusions invalidated | It used padded 8x8/192 mini tensors |
| Job 33517, standard soft-Z model | `bootstrap-v0` only | Small screen and overlapping uncertainty do not select an architecture/target winner |
| Jobs 33522–33531, neural self-play pilots | Throughput/diagnostic evidence | Tiny pilots cannot establish four-way averaging, no-noise, or a simulation optimum |
| Jobs 33536+, first neural replay | Regression evidence, not progress | The replay window did not follow the documented pretraining seed plan; forgetting remains plausible |

## Experiment ledger

| Phase / jobs | Board | Purpose | Status | Promotion value |
| --- | --- | --- | --- | --- |
| 33475 | 8x8 boundary | TensorFlow GPU train/save/load smoke | Passed on earlier code | Environment proof only; rerun native boundary |
| 33478–33479 | 5x5 | Baseline paired tournament | Passed | Arena and baseline Elo plumbing |
| 33486–33502 | 5x5 | Dummy self-play/search pilots | Passed | Reject disasters; data generator starting point |
| 33494–33511 | 5x5 padded | Fixed-data learner | Archived | Loss plumbing only |
| 33516 | 5x5 padded | Neural arena | Archived | No native architecture claim |
| 33517 | 8x8 | First standard neural screen | Preliminary | Names `bootstrap-v0`; requires broader data and confirmation |
| 33518–33525 | 8x8 | Neural batching | Passed | Batched independent games retained |
| 33522–33531 | 8x8 | Neural self-play diagnostics | Preliminary | Informs hypotheses, not defaults |
| 33538 | 5x5 and 8x8 | Native TensorFlow, masking, save/load, cross-rule rejection | Passed | Authorizes native mini experiments |
| 33539 | 5x5 | Initial `c_puct` clock screen | Rejected | 3--6 time forfeits per task; diagnostic only |
| 33543 | 5x5 | Revised 50 ms `c_puct` screen | Partial | 0.25/1.5 clean; retry 0.75/3.0 after one scheduler overrun each |
| 33547 | 5x5 | Final-protocol retry for 0.75/3.0 | Passed | Clean screen advances 1.5 and 3.0 to direct confirmation |
| 33549 | 5x5 | Direct 1.5 vs 3.0 confirmation | Passed | 90-102, CI overlaps zero; preregistered rule retains 1.5 |
| 33550/33558 | 5x5 | 512-game corpus and postflight | Passed | 6,838 positions; 512 unique trajectories; all checks valid |
| 33559/33565 | 5x5 | Six native learners and checkpoint selection | Passed | Hash-verified early-stopping choices ready for Elo screen |
| 33566 | 5x5 | First native-model screen wrapper | Rejected | JQ/`readonly` path bug; zero games and negligible allocation |
| Native fixed-data screen | 5x5 | 2 trunks × 3 value targets on expanded raw data | Preregistered, not submitted | Selects candidates only after fresh Elo |

## Next gated actions

1. Generate an independent 512-game mini rollout-MCTS stage in parallel CPU
   shards; expand only if learning curves remain data-limited.
2. Train `32x3` and `64x4` native models for outcome, soft-Z, and the fixed
   mixture on identical game splits and wall-clock budgets.
3. Run fresh paired arenas with model hashes and uncertainty. Use tiny matches
   only to reject failures, then confirm survivors.
4. Run the finite 5x5 search/exploration plan from the literature survey.
5. Freeze one coherent baseline and at most three variants, obtain a fresh
   three-risk audit, then perform narrow 8x8 confirmation.

No expanded standard self-play or online neural replay is authorized by this
status page yet.
