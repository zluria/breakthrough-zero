# First standard-board neural screening: job 33517

This 8x8 screen crossed two dummy-MCTS data budgets (32 or 64 simulations)
with two absolute value targets (final outcome or soft-Z).  All four models used
the same 64-channel, four-block CNN, exactly 2,500 training and 600 validation
positions, 40 epochs, data split, augmentation procedure, and optimizer.

Every matchup used the same eight saved four-ply openings with color reversal,
for 16 games.  Rated search had no noise, received 50 ms per move, and was
checked with an external 100 ms scheduler grace that was not available to the
agent.  All **448 games ended at a rules terminal**.  There were no time
forfeits, illegal moves, agent errors, draws, malformed pairs, or failed seed
reversals.  Raw evidence is in
[`results/standard-neural-33517`](../../results/standard-neural-33517/README.md).

## Global screening ratings

| Agent | Fitted Elo |
| --- | ---: |
| Alpha-beta | 1791.8 |
| Tactical-rollout PUCT | 1492.5 |
| 64-simulation soft-Z network | 1428.8 |
| 64-simulation outcome network | 1406.5 |
| Plain-rollout PUCT | 1383.6 |
| 32-simulation soft-Z network | 1291.4 |
| 32-simulation outcome network | 1225.6 |
| Random | 1000.0 |

These pool ratings are a compact screen; the eight-pair head-to-head intervals
below are the evidence for design choices.

## Controlled comparisons

| First agent vs second | W-L-D | Elo difference | 95% CI |
| --- | ---: | ---: | ---: |
| 64-sim soft-Z vs 64-sim outcome | 12-4-0 | +166.0 | [-69.1, +401.0] |
| 32-sim soft-Z vs 32-sim outcome | 7-9-0 | -38.8 | [-253.3, +175.7] |
| 64-sim outcome vs 32-sim outcome | 16-0-0 | +492.2 | [+91.6, +892.8] |
| 64-sim soft-Z vs 32-sim soft-Z | 9-7-0 | +38.8 | [-175.7, +253.3] |
| 64-sim soft-Z vs plain-rollout PUCT | 10-6-0 | +78.5 | [-139.7, +296.7] |
| 64-sim soft-Z vs tactical-rollout PUCT | 7-9-0 | -38.8 | [-253.3, +175.7] |
| 64-sim soft-Z vs alpha-beta | 0-16-0 | -492.2 | [-892.8, -91.6] |

Soft-Z is not a universal Elo win in this small screen: it led 12-4 on the
64-simulation data but trailed 7-9 on the 32-simulation data.  Its much better
held-out value error and its repeated mini-board advantage still make it the
more useful bootstrap target.  The result remains preliminary.

The extra MCTS work used to create the 64-simulation dataset had a large,
statistically clear effect for outcome training, but only a 9-7 effect under
soft-Z.  We should not infer that doubling simulations always pays: data
generation throughput fell from roughly 30 to 18 positions/second, and the
target interaction is real.

## Timing and work audit

| Agent | Moves | Mean elapsed (s) | 95th percentile (s) | Mean work units |
| --- | ---: | ---: | ---: | ---: |
| Alpha-beta | 2198 | 0.04566 | 0.05009 | 2900.4 nodes |
| Tactical-rollout PUCT | 2711 | 0.05319 | 0.09441 | 197.1 simulations |
| Plain-rollout PUCT | 2471 | 0.05301 | 0.08989 | 179.5 simulations |
| 32-sim outcome network | 2935 | 0.04575 | 0.04645 | 4.1 simulations |
| 32-sim soft-Z network | 3008 | 0.04578 | 0.04643 | 4.1 simulations |
| 64-sim outcome network | 2831 | 0.04538 | 0.04644 | 4.3 simulations |
| 64-sim soft-Z network | 2890 | 0.04570 | 0.04632 | 4.4 simulations |

The CNN agents receive only about four complete simulations per rated move.
This is the dominant engineering bottleneck and makes batch-one search a poor
regime in which to tune final PUCT constants.  Long-tail CPU records again had
low work counts consistent with cluster descheduling rather than extra search.

## Decision

Promote the 64-simulation soft-Z checkpoint as the provisional 8x8 bootstrap
model, while retaining every alternative checkpoint and raw game.  It passed
the random sanity check 16-0, was competitive with both rollout PUCT agents,
and combines the best pool rating with the best held-out value diagnostics.

Do not begin a large self-play run yet.  First measure GPU inference batching,
then run the scalar-equivalent batched actor gate.  Alpha-beta's sweep remains
the immutable strength target that self-play must eventually overcome.
