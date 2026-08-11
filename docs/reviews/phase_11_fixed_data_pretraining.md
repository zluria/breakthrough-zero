# Phase 11 review: fixed-data mini pretraining

## Question

Can the policy/value network learn useful search targets from the same small,
verified dataset, and do model size or value target change that answer?

The experiment is a 2x2 design:

| Trunk | Final-result value | Soft-Z value |
| --- | --- | --- |
| 32 channels, 3 residual blocks | task 0 | task 1 |
| 64 channels, 4 residual blocks | task 2 | task 3 |

All tasks use the 64 tactical-rollout games generated with 32 simulations per
move.  They share the exact game-level train/validation split, seed, batch
size, optimizer, loss weights, augmentation rule, and epoch count.

## Why these four jobs are justified

The self-play pilot suggests tactical rollouts improve the root value at little
or no throughput cost.  Thirty-two simulations produced the highest normalized
visit entropy, so it is the best first dataset for asking whether a network can
learn more than the most-visited action.

The smaller trunk tests whether the baseline is overbuilt for 5x5.  The larger
trunk is the provisional 8x8 architecture.  Final result is the assignment's
simple reference target; soft-Z is the literature-motivated contender.  A full
factorial avoids accidentally attributing an architecture interaction to the
value target.

## Predictions registered before submission

1. Soft-Z will show lower value error because its continuous targets are less
   extreme; raw losses across the two targets are therefore not directly a
   strength comparison.
2. The 64x4 trunk will fit training data faster, but may not improve held-out
   policy loss on only 64 games.
3. Policy learning should be visible in both sizes.  Failure to beat the
   initial approximately uniform legal-policy loss is a stop condition.
4. These training curves will select candidates for Elo testing, not declare a
   winner.  Playing evaluation remains paired and equal wall-clock.

## Augmentation and loss

Every training draw independently selects one of the four exact Breakthrough
symmetries.  Validation uses identity examples for a stable curve.  Policy
cross-entropy is computed after masking illegal actions; unmasked illegal mass
is diagnostic only and is not expected to vanish under this loss.  The value
target and output remain absolute Player-1 values after augmentation.

## Budget and stop conditions

Each task receives one RTX3070, two CPUs, four GB, and at most 15 minutes.  Two
tasks may run concurrently.  Thirty epochs are cheap on roughly 700 training
positions and are intended to reveal fitting behavior, not to establish an
online-training schedule.

Stop if a checksum, replay, finite-loss, GPU, or save/load gate fails.  Do not
generate more self-play based only on these training losses.
