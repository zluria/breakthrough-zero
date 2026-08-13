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
