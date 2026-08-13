# Second native 5x5 learning cycle: jobs 33607--33609

This cycle repeated the generation-1 protocol with a new parent and seeds,
while holding architecture, search, replay loss ratio, value target, training
time, and arena time fixed.

## Data and training

- Parent: generation 1 epoch 28,
  `1c68f6cd159ed6cd273f9703bfb2ff1848d0c2dde5fae13e7a0e786263781ce3`.
- New self-play: 256 games, 3,371 positions, 243 unique trajectories.
- Train/validation: 615/153 complete games with exact 75/25 rollout/neural
  loss shares.
- Selected child: epoch 4,
  `fcab56f571c1392443871d6b012c2307ed42b493ed88c2fe3d7a4e35b1501d4d`.

The actual validation minimum was epoch 3 (2.28450), but the four-epoch
periodic checkpoint schedule had not saved it. Epoch 4 scored 2.28588. This
small miss did not explain an Elo collapse, but exposed a real rollback flaw:
short fine-tunes need best-so-far checkpoints and the unchanged parent as an
explicit epoch-0 candidate.

## Fresh arena

All 640 games completed normally on 64 new color-reversed opening pairs per
matchup.

| First agent vs second | W-L | Elo [95% CI] |
| --- | ---: | ---: |
| Generation 1 vs generation 2 | 65-63 | +5 [-78, +89] |
| Alpha-beta vs generation 1 | 61-67 | -16 [-100, +68] |
| Alpha-beta vs generation 2 | 65-63 | +5 [-78, +89] |
| Tactical PUCT vs generation 1 | 66-62 | +11 [-73, +94] |
| Tactical PUCT vs generation 2 | 67-61 | +16 [-68, +100] |

Equal-time work was comparable: both neural agents had median 4 and p90 10
work units per move, and mean move time was 38.8 ms. Generation 2 is a plateau,
not the familiar catastrophic self-play regression. Continuing to generation
3 with unchanged inputs would spend compute without evidence. The likely
bottlenecks are the dominant static rollout anchor, limited new policy support,
or imperfect correlation between held-out imitation loss and playing strength;
these are hypotheses for controlled fixed-data tests, not post-hoc conclusions.
