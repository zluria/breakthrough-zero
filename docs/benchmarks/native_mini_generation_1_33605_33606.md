# First native 5x5 learning cycle: jobs 33605 and 33606

The complete path `generation 0 -> neural self-play -> replay training ->
generation 1 -> fresh Elo` ran successfully at commit `7a8bede`. This is the
first trustworthy cycle because both networks use the native 5x5 input and
75-action policy boundary.

## Training evidence

- Rollout corpus: 512 games, 6,838 positions.
- Parent-generated corpus: 256 games, 3,269 positions.
- Independent complete-game splits: 615 train games / 153 validation games.
- Exact loss shares in each partition: 75% rollout / 25% neural.
- Initial model: generation 0 soft-Z epoch 84,
  `3bc3ca17393ce803d3d89ccc02f649cfb755b56b1af944c1f41c82db4114beb6`.
- Selected model: epoch 28,
  `1c68f6cd159ed6cd273f9703bfb2ff1848d0c2dde5fae13e7a0e786263781ce3`.
- Selection objective: minimum held-out weighted total, 2.30625.

The 120-second run continued well past epoch 28 and validation later worsened.
Taking the last epoch would therefore have been an avoidable selection bug.

## Fresh paired arena

The arena used 64 new four-ply openings per matchup, replayed with colors
reversed, 50 ms per move, no search noise, and no symmetry ensemble. All 640
games ended normally.

| First agent vs second | W-L | Elo [95% CI] |
| --- | ---: | ---: |
| Generation 0 vs generation 1 | 55-73 | -48 [-133, +36] |
| Alpha-beta vs generation 0 | 72-56 | +43 [-41, +127] |
| Alpha-beta vs generation 1 | 58-70 | -32 [-116, +52] |
| Tactical PUCT vs generation 0 | 81-47 | +93 [+6, +180] |
| Tactical PUCT vs generation 1 | 74-54 | +54 [-31, +139] |

The direct confidence interval crosses zero, so one seed cannot establish a
general strength gain. The child nevertheless beat its parent in the observed
games and narrowed both anchor gaps. Under the preregistered rule this is a
non-regressing result, not a promoted research conclusion, and it authorizes
one repeat cycle with all major choices frozen.
