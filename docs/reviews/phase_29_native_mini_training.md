# Phase 29 review: native-mini architecture and value targets

## Question

On the same audited 512-game corpus, which of two small native CNN trunks and
three simple absolute value targets produce plausible learning curves and
playing candidates under a limited learner budget?

| Trunk | Outcome | Root Q (soft-Z) | 50:50 outcome/root-Q |
| --- | --- | --- | --- |
| 32 channels, 3 blocks | task 0 | task 1 | task 2 |
| 64 channels, 4 blocks | task 3 | task 4 | task 5 |

This full factorial prevents a target/architecture interaction from being
misreported as a universal target result. It is still a screen: six jobs are
enough, and adding more heads or trunks before fresh Elo would be unjustified.

## Fairness and augmentation

Every task uses the identical game-level train/validation split, seed, batch
size, learning rate, loss weights, validation data, and 120-second learner
budget. Actual optimizer examples and elapsed time are recorded. Larger trunks
may process fewer examples in the fixed time; that is part of their compute
cost, not something hidden by equal epoch counts.

Each original training position sees all four exact symmetries in a balanced
four-epoch cycle. Validation remains identity-only. The target and output are
always from absolute Player 1's point of view. The policy target is the same
stored visit distribution in all six tasks.

Periodic checkpoints are saved every four epochs and again at the stopping
epoch. This bounds storage while retaining a learning curve and avoids judging
only the final model. Checkpoint hashes, data hashes, Git commit, GPU identity,
metrics, sample counts, and elapsed time are recorded.

## Interpretation and stop rules

- Loss scales differ across value targets; lower total loss does not establish
  a stronger agent.
- Reject non-finite training, worsening policy calibration, gross train/
  validation divergence, or a value head collapsed to one side.
- Use curves to choose a small number of checkpoints for fresh paired arenas;
  do not repeatedly arena-test every epoch and select on noise.
- All playing comparisons use identical 50 ms search clocks, paired four-ply
  openings, color reversal, no search noise, model hashes, and uncertainty.
- Explicitly compare the trained agents to the rollout-MCTS teacher and fixed
  alpha-beta anchor. A model that beats only its predecessor has not escaped
  the warned failure mode.

Do not start neural self-play from these models until one is demonstrably
competitive with the pretrained baseline and its value/policy diagnostics are
sound. If all six regress, diagnose the shared data/optimization contract
before generating more games.
