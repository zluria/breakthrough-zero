# Replay buffer: recency, diversity, and limited compute

## The tradeoff

A small window is current but highly correlated. It can overfit, forget older
skills, and amplify a self-play blind spot. A large window is diverse but stale;
weak early policies can slow improvement long after the agent has surpassed
them. Disk capacity is not the main constraint here. The important quantities
are sample age and how often each expensive position is reused.

Published systems span very different regimes. AlphaGo Zero and ELF OpenGo
used very large 500,000-game buffers. A resource-efficient Hex system reported
that a 2-million-position window--only 64 of its very large environment
steps--worked better because it reduced staleness. AlphaZero.jl recommends a
growing buffer and used 200,000 to 1 million positions for Connect Four, about
6 to 29 new-data batches. These numbers are evidence that scale matters, not
portable Breakthrough constants.

Sources:

- [ELF OpenGo supplement](https://proceedings.mlr.press/v97/tian19a/tian19a-supp.pdf)
- [Scaling Scaling Laws with Board Games](https://arxiv.org/pdf/2104.03113)
- [AlphaZero.jl training parameters](https://jonathan-laurent.github.io/AlphaZero.jl/v0.3/reference/params/)
- [AlphaZero.jl Connect Four run](https://jonathan-laurent.github.io/AlphaZero.jl/v0.3/tutorial/connect_four/)

A recent practitioner discussion suggests keeping typical reuse in the single
digits and debugging training on a fixed dataset before reconnecting the
self-play feedback loop. We already follow the second recommendation; the first
is a pilot range, not a fact.

Source:

- [Practitioner discussion of AlphaZero data reuse](https://www.reddit.com/r/MachineLearning/comments/1tvw6sc/analysis_of_alphazero_training_data_d/)

## Our design

All chunks remain in the immutable archive. The **active training window** is a
FIFO view over the newest complete games, capped by recorded positions. Games
are never split across train and validation. Data is sampled uniformly by
recorded position in the baseline; recency weighting is a later one-factor
ablation.

The online window is initially seeded with the newest pretraining games and
then naturally evicts them. This avoids a sudden distribution jump without
letting rollout-MCTS data dominate forever. Pretraining architecture studies
continue to use the separate fixed archive.

The buffer does not silently determine optimization volume. We log:

- Capacity and actual positions/games.
- Position age in wall time and in model checkpoints.
- Fraction from the current and each older model.
- New positions per wall-clock hour.
- Optimizer examples per new position (reuse factor).
- Unique-state and opening redundancy.
- Policy surprise and value calibration by age bucket.
- Recent-data and fixed-anchor validation losses.

## Choosing the capacity

The HPC pilot first measures recorded positions per hour. We then freeze three
position capacities corresponding approximately to short, medium, and long
generation histories on that hardware. A reasonable initial normalized range
is about 8, 24, and 64 batches of newly generated positions; exact capacities
are chosen only after throughput and average game length are known.

The first comparison uses short equal-wall-clock online runs and one seed to
discard clear failures. The surviving two settings are repeated. The primary
metric is Elo improvement per end-to-end hour; secondary diagnostics reveal
whether a small buffer overfits or a large one is stale.

We will not tune capacity by validation loss alone. Old-window validation can
reward remembering obsolete play, while recent-only validation can reward
collapse. Ratings against immutable pure-MCTS, pretrained-PUCT, alpha-beta, and
older neural anchors are required.

## Growth and overtraining safeguards

The learner waits for a minimum diverse window rather than training on the
first handful of games. Capacity may grow from the short to selected final
window so early weak data washes out quickly. Optimizer work is controlled by a
target reuse range and wall-clock allocation; a small buffer is not an excuse
to run unlimited epochs over correlated positions.

Random symmetry is chosen in the loader on each draw from the four exact
augmentations. This avoids placing four adjacent, highly correlated copies in
the replay buffer while preserving all four transformations over time.
