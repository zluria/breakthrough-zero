# Fixed-data Keras pretraining

All training used game-level splits, one randomly chosen exact symmetry per
training draw, identity validation, Adam at `0.001`, equal policy/value loss
weights, and native Keras checkpoints.  Values remained absolute Player-1
values under every transformation.

## Mini board: job array 33494

The four models saw the same 64 tactical-rollout games generated with 32
simulations.  Final validation metrics after 30 epochs:

| Model | Value target | Policy CE | Policy top-1 | Value MSE | Value MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| 32 channels x 3 blocks | Outcome | 2.227 | 0.267 | 1.034 | 0.998 |
| 32 channels x 3 blocks | Soft-Z | 2.228 | 0.267 | 0.220 | 0.369 |
| 64 channels x 4 blocks | Outcome | 2.241 | 0.279 | 1.009 | 0.998 |
| 64 channels x 4 blocks | Soft-Z | 2.237 | 0.230 | 0.123 | 0.281 |

The final outcome was essentially unpredictable from this small dataset,
while soft-Z generalized.  Architecture did not clearly improve policy loss.
The first neural arena intended to test strength was later rejected because of
time forfeits; see [`mini_neural_33498_invalid.md`](mini_neural_33498_invalid.md).

## Standard board: job array 33511

Every model used exactly 2,500 training and 600 validation positions.  The
64-channel, four-block architecture, seed, split procedure, batches, and 40
epochs were fixed.

| Search data | Value target | Policy CE | Policy top-1 | Value MSE | Value MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 simulations | Outcome | 2.559 | 0.190 | 1.006 | 1.003 |
| 32 simulations | Soft-Z | 2.569 | 0.175 | 0.156 | 0.309 |
| 64 simulations | Outcome | 3.112 | 0.143 | 1.302 | 0.954 |
| 64 simulations | Soft-Z | 3.105 | 0.127 | 0.104 | 0.235 |

Soft-Z again provided a much more learnable value signal.  Raw policy
cross-entropy cannot compare the 32- and 64-simulation datasets fairly because
their target entropies differ substantially.  The learner now reports target
entropy and `cross_entropy - entropy` (policy KL) for future runs.  Paired Elo
is still the promotion criterion.

Each standard job completed in 85--88 seconds and each checkpoint directory is
about 1.3 MB.  The full raw data and checkpoints remain under the durable HPC
data root with input checksums and run manifests; the promoted checkpoints will
be attached to a GitHub release after valid arena selection.
