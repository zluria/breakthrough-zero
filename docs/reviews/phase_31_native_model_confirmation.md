# Phase 31 review: confirm the two native-mini finalists

## Question and candidates

The 24-pair failure screen advanced two coherent candidates:

- 32 channels, 3 residual blocks, soft-Z, epoch 84;
- 64 channels, 4 residual blocks, soft-Z, epoch 56.

The smaller model was closest to tactical rollout PUCT; the larger was closest
to alpha-beta. Their screen intervals overlap, so fitted Elo from separate
tasks cannot choose between them.

## Fair confirmation

Load both hash-verified checkpoints into one GPU task. On 64 entirely new
four-ply opening pairs, play each model against alpha-beta and tactical-rollout
PUCT and play the models directly. Every opening is used once with each color.
The harness excludes the irrelevant alpha-beta vs tactical-PUCT repeat, leaving
five matchups and 640 total games.

Every search receives 50 ms internally, with the already validated 100 ms
external scheduler grace. `c_puct=1.5`; search noise and symmetry averaging are
off. Model architecture cost therefore affects simulations completed under the
same clock, which is the intended strength-per-inference-time comparison.

## Decision rule

- Any abnormal game invalidates the run.
- Prefer a model that wins their direct match and is not clearly worse against
  either fixed anchor.
- If the direct interval overlaps zero, prefer the smaller 32x3 model unless
  the larger model shows a material anchor advantage. This is an explicit
  simplicity/throughput rule, not a post-hoc tie break.
- This chooses a pretraining baseline, not permission for neural self-play.
  First diagnose policy/value calibration and confirm that the chosen model is
  not clearly below the rollout teacher.

Do not add more targets or opening seeds to force a winner. The next experiment
must address a new mechanism such as exploration, search batching, or global
pooling under its own preregistered budget.
