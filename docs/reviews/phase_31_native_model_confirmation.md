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

## Result

Slurm job 33584 completed all five matchups on RTX3070-08 in 3 minutes 57
seconds. The postflight found zero failures and 640 terminal games. Each row
below used 64 fresh opening pairs (128 games); Elo and its 95% interval are
from the first-named agent's perspective.

| Matchup | W-L | Elo [95% CI] | Pair sweeps, first-second |
| --- | ---: | ---: | ---: |
| Alpha-beta vs 32x3 soft-Z | 67-61 | +16 [-68, +100] | 10-7 |
| Alpha-beta vs 64x4 soft-Z | 71-57 | +38 [-47, +122] | 15-8 |
| Tactical PUCT vs 32x3 soft-Z | 78-50 | +76 [-10, +162] | 19-5 |
| Tactical PUCT vs 64x4 soft-Z | 76-52 | +65 [-20, +150] | 20-8 |
| 32x3 soft-Z vs 64x4 soft-Z | 57-71 | -38 [-122, +47] | 4-11 |

The larger model won the direct point estimate, but its interval overlaps zero.
It was 22 Elo points worse against alpha-beta and 11 points better against
tactical PUCT; neither is a material anchor advantage. Across all five files,
the smaller model completed a mean 19.4 search work units per move versus 11.5
for the larger model under the same clock. No model exceeded the 150 ms
classification boundary.

## Decision

Select **32x3 soft-Z, epoch 84** as the coherent native-mini baseline. This
follows the written tie rule: the direct result is unresolved, the anchors do
not provide a clear override, and the smaller network is simpler and searches
more nodes per unit time. Its checkpoint SHA-256 is
`3bc3ca17393ce803d3d89ccc02f649cfb755b56b1af944c1f41c82db4114beb6`.

This is deliberately a modest conclusion. All six learners used one raw
corpus, split, and training seed. The screen advanced only soft-Z models, but
its intervals overlap; it does not establish a universal value-target winner.
The selected neural agent also trailed tactical PUCT 50-78 by point estimate,
with an interval just crossing zero. Neural self-play remains gated on
calibration diagnostics and a separately preregistered mechanism experiment.
