# Phase 35: one-hour large-data diagnosis

## Decision question

Generation 2 plateaued after learning from only 256 current-policy games. Before
changing replay, noise, search, targets, or architecture, test whether a much
larger sample from the same policy makes the existing loop move.

This is deliberately an end-to-end resource test rather than another small
screen. The expected allocation is about one hour on one RTX 3070.

## Frozen choices

- Parent: generation 1 epoch 28, SHA-256
  `1c68f6cd159ed6cd273f9703bfb2ff1848d0c2dde5fae13e7a0e786263781ce3`.
- Self-play: 32 simulations, `c_puct=1.5`, parent-Q FPU, sampling through ply
  4, no root noise, batch 64.
- Training: native 32x3 CNN, `mixed_z_q`, Adam `3e-4`, batch 256, exact 75/25
  rollout/neural loss, and 120 seconds. The unchanged parent is epoch 0; every
  new validation best is recoverable.
- Evaluation: noise-free 50 ms search, color-reversed four-ply openings, and
  the same alpha-beta and tactical-PUCT anchors.

## Increased budgets

Generate 12,288 games with the same master seed as the earlier 256-game batch.
The small batch is therefore a deterministic prefix of this separately stored,
single-commit corpus. This is 48 times as many games and should contain roughly
160,000 positions. The learner still receives the same 120 seconds: larger
data therefore means less reuse, not a hidden increase in optimizer time.

Use 256 fresh opening pairs per arena matchup, four times the previous arena.
The direct confidence interval should narrow from roughly +/-84 Elo to about
+/-42 Elo if game variance is similar.

## Interpretation fixed before launch

- If the large-data child beats generation 1 and moves favorably against the
  anchors, the 256-game batch was too small. Next choose a sustainable buffer
  capacity before altering the algorithm.
- If it rolls back to epoch 0 or remains flat, more noise-free games at a 25%
  loss share are not the immediate bottleneck. Use the saved corpus for a
  replay-ratio experiment before generating again.
- If it regresses clearly, stop. Inspect recent-versus-pretraining validation,
  value calibration, policy coverage, and optimizer displacement; do not tune
  around the result.

This experiment does not determine the best replay ratio. Keeping 75/25 fixed
is what makes it a useful test of data quantity.

## Result

Job 33611 completed in 45 minutes. It generated 12,288 games and 163,527
positions from the pinned generation-1 model. The archive contains 6,821
distinct trajectories and 21,333 distinct positions. All data checks passed.

Training presented 953,211 examples over seven epochs. The unchanged parent
had validation loss 2.2918; every trained epoch was worse (2.3155 to
2.3324). Checkpoint selection therefore returned epoch 0. Under the frozen
decision rule, this means **no model improvement was selected**.

The 75/25 source objective also exposed an important learner problem. Only
about 4% of raw training positions came from pretraining, so aggregate source
weighting gave each pretraining position about 71 times the weight of a neural
self-play position. The nominal effective sample size of a 256-position batch
was only about 18. This is a strong candidate explanation for unstable
fine-tuning, not yet a proven cause.

## Evaluation incident

The job incorrectly continued into the arena after selecting epoch 0. The
“generation-1” and “large-data” labels pointed to the same SHA-256 checkpoint.
There were 256 distinct opening positions, each played with colors reversed;
because both agents were identical and deterministic, the two games in every
pair followed the same board trajectory with the labels exchanged. The exact
256-256 score is therefore an invalid self-comparison and contains no Elo
evidence.

The tournament driver now rejects duplicate `(checkpoint hash, inference
mode)` identities before creating output, and the Slurm wrapper exits when a
selection returns the parent hash. This is deliberately a small invariant,
not another promotion framework.

## Decision

Do not generate more self-play yet. Preserve the 12,288-game archive and use
it for a fixed-data study of the training objective and replay sampling. Any
future arena must compare a trained checkpoint with a different hash; a
rollback is reported as “no candidate,” not evaluated as a new agent.
