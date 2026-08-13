# Phase 34: repeat the native learning cycle

## Question

Does the first favorable cycle repeat when generation 1 produces a new,
independently seeded self-play corpus? This is a pipeline-reliability check,
not a parameter sweep.

## Frozen protocol

- Parent: generation 1 epoch 28, SHA-256
  `1c68f6cd159ed6cd273f9703bfb2ff1848d0c2dde5fae13e7a0e786263781ce3`.
- Generate 256 native 5x5 games with 32 simulations, `c_puct=1.5`, parent-Q
  FPU, visit sampling through ply 4, and no root noise.
- Preserve generation 0's corpus, but train this cycle on the fixed rollout
  anchor plus the newest 256 neural games. Keeping the active window size fixed
  prevents a growing dataset from silently changing the compute experiment.
- Keep the 75/25 rollout/neural loss split, `mixed_z_q`, 32x3 CNN, Adam at
  `3e-4`, batch 256, four-symmetry cycle, 120-second limit, and held-out
  checkpoint selection.
- Evaluate generation 2 against generation 1 and both fixed anchors on 64 new
  color-reversed opening pairs at 50 ms per move.

Self-play seed is `202608133501`, training seed `202608133601`, and arena seed
`202608133701`. Any audit failure or abnormal arena game invalidates the run.
If generation 2 is clearly worse, stop the loop and diagnose. If the direct
interval overlaps zero, require favorable or neutral anchor movement before
continuing.

## Result and stop decision

Jobs 33607--33609 completed at commit `814035c`. Self-play produced 256 valid
games and 3,371 positions; 243 trajectories were unique. Training selected
epoch 4, SHA-256 `fcab56f571c1392443871d6b012c2307ed42b493ed88c2fe3d7a4e35b1501d4d`.
All 640 arena games ended normally.

| Matchup | W-L | Elo difference [95% CI] |
| --- | ---: | ---: |
| Generation 1 vs generation 2 | 65-63 | +5 [-78, +89] |
| Alpha-beta vs generation 1 | 61-67 | -16 [-100, +68] |
| Alpha-beta vs generation 2 | 65-63 | +5 [-78, +89] |
| Tactical PUCT vs generation 1 | 66-62 | +11 [-73, +94] |
| Tactical PUCT vs generation 2 | 67-61 | +16 [-68, +100] |

Generation 2 is statistically and practically tied with its parent. Both
anchor movements are small and unfavorable, so it is non-regressing under the
declared interval rule but provides no evidence of continued learning. The
loop stops here; generation 3 is not authorized.

The diagnosis found no wall-clock evaluation imbalance: both neural agents had
median 4 and 90th-percentile 10 search work units per move, with mean move time
38.8 ms. The stronger signal is optimization saturation. The true minimum
validation total occurred at epoch 3, immediately after initialization, and
later training overfit. Because checkpoints were periodic every four epochs,
epoch 3 was not recoverable. The learner must preserve every new validation
best and treat the unchanged parent as an epoch-0 rollback candidate before
the next learning experiment.
