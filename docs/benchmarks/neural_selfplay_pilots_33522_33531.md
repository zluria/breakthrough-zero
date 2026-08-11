# Neural self-play diagnostics: jobs 33522--33531

These pilots used the selected standard-board 64x4 soft-Z checkpoint
(`f070b3a8...959999`) and immutable schema-3 output. Unless a row says
otherwise, move selection sampled root visits through ply 12, root noise was
off, `c_puct` was 1.5, and all 64 games used the same recorded seeds.

## Structural checks before tuning

- The complete 80-test suite passed on the HPC, including the hand-calculated
  absolute-P1 PUCT tree, terminal moves that do not switch turn, legal policy
  masks, all four data symmetries, and scalar-versus-batched games.
- Every saved chunk was reloaded and checked against its manifest, model hash,
  configuration, game-index interval, seeds, and checksum.
- An audit found that the loader repeatedly decompressed NPZ arrays from inside
  the action loop. Materializing each compressed member once reduced one
  2.6 MB chunk reload from roughly 109 seconds to 0.87 seconds without changing
  data. This was fixed before interpreting end-to-end throughput.

## Model symmetry diagnosis

Job 33524 evaluated 256 exact transformed position pairs. The pretrained model
was not close to invariant despite correct, balanced augmentation:

| Transform | Mean policy L1 | Top-move agreement | Mean value residual |
| --- | ---: | ---: | ---: |
| Left-right mirror | 0.379 | 0.242 | 0.0916 |
| Swap players | 0.385 | 0.262 | 0.1575 |
| Swap + mirror | 0.229 | 0.438 | 0.1609 |

For player-swapping transforms, the residual compares one absolute-P1 value
with the negative of the transformed value. The training-balance audit then
confirmed that every selected example appeared under all four symmetries over
40 epochs, symmetry draws were balanced, and the fully augmented soft-Z mean
was exactly zero. The defect was learned approximation/underfitting, not a
missing transform or an accidental relative-value conversion.

`SymmetryEnsembleEvaluator` therefore averages policies over all four exact
transforms, maps every move back, and negates values only when a transform
swaps player labels. Public values and all tree Q values remain absolute P1.
At 64 concurrent games, one ensemble call contains 256 states and retains
about 2,581 effective leaves/second on the RTX 3070.

## Pilot results

| Job | Evaluator | Sims | Sample plies | Positions | Mean plies | P1 wins | Visit entropy | Top share | Unique positions | Chunk seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 33522 | Raw network | 16 | 12 | 4,057 | 63.39 | 9/64 | 0.741 | 0.202 | 96.9% | not comparable¹ |
| 33526 | Four-symmetry ensemble | 16 | 12 | 4,294 | 67.09 | 18/64 | 0.770 | 0.175 | 97.1% | 65.03 |
| 33527 | Ensemble, zero value | 16 | 12 | 4,610 | 72.03 | 20/64 | 0.798 | 0.166 | -- | 68.57 |
| 33528 | Ensemble, uniform policy | 16 | 12 | 2,768 | 43.25 | 0/64 | 0.805 | -- | -- | 51.56 |
| 33529 | Four-symmetry ensemble | 32 | 12 | 4,549 | 71.08 | 27/64 | 0.875 | 0.149 | 97.6% | 134.69 |
| 33530 | Four-symmetry ensemble | 64 | 12 | 4,434 | 69.28 | 28/64 | 0.900 | 0.150 | 97.6% | 261.78 |
| 33531 | Four-symmetry ensemble | 32 | 4 | 4,719 | 73.73 | 29/64 | 0.875 | 0.151 | 97.7% | 134.24 |

¹ Job 33522 used the old slow reload path, so its printed chunk duration mixes
generation with more than 100 seconds of redundant decompression.

All seven runs produced 64 unique trajectories and 64 unique four-ply
prefixes, including every run with no Dirichlet noise. For raw versus ensemble
in the paired 16-simulation runs, 16 seeds changed from a Player-2 to a
Player-1 win and seven changed in the opposite direction; an exact McNemar
test is still inconclusive (`p` about 0.093). For full ensemble versus zero
value, the corresponding changes were 13 and 11. That ablation provides no
evidence that the value head caused the remaining seat skew.

The uniform-policy run is a deliberately preserved failed diagnostic, not a
playing-strength conclusion. Standard Breakthrough has roughly 22 root moves,
so a 16-simulation search cannot even visit every equal-prior child. Legal-move
iteration order then dominates before the value head can compare all choices.
The run usefully exposed that 16 simulations is below the structural opening
budget.

## Decisions

1. Use exact four-symmetry inference averaging for this bootstrap generation.
   Continue random symmetry augmentation in training; the ensemble is a
   correctness/stability boundary, not a substitute for learning symmetry.
2. Use 32 simulations for the first neural replay batch. It normally clears
   the opening branching factor and generated about 34 positions/second.
   Sixty-four simulations halved throughput for only small aggregate changes.
3. Sample root visits through ply 4. Job 33531 retained complete prefix and
   trajectory diversity and had the best small-sample seat balance.
4. Keep Dirichlet root noise off. Ordinary visit sampling already supplies
   diverse games; add noise later only in response to measured collapse.
5. Generate only a bounded 256-game replay batch next, split into two
   independent 128-game jobs. Train and evaluate on that fixed data before
   authorizing a longer loop.

These are provisional, resource-specific engineering decisions. Aggregate
self-play color balance and target entropy are diagnostics, not Elo. Playing
strength still requires paired, color-reversed, equal-wall-clock evaluation.
