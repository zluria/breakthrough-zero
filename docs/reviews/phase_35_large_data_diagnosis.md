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
