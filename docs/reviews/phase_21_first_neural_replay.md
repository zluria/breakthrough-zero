# Phase 21 review: first bounded neural replay batch

## Decision entering this phase

The production candidate uses exact four-symmetry inference averaging, 32
simulations per move, `c_puct=1.5`, visit sampling through ply 4, and no root
noise. The four-ply pilot produced 64 distinct trajectories and 64 distinct
four-ply prefixes, 97.7% unique positions, and a 29--35 color split. Extra
Dirichlet noise therefore has no demonstrated benefit at this point.

Thirty-two simulations are the smallest tested budget that normally visits
every opening move. Compared with 64 simulations, it approximately doubles
position throughput while the pilot target statistics and color balance were
similar. This is a resource-aware choice, not a claim that 32 is universally
optimal.

## Bounded run

Generate two independent 128-game jobs on separate RTX 3070 GPUs. Each job
writes two immutable 64-game chunks with a distinct recorded master seed.
Together this should yield roughly 18,000 raw positions. It is deliberately a
small first replay generation, not an arbitrary commitment to 10,000 games.

The two-job design has three useful properties:

1. it fills two GPUs without coordinating mutable state;
2. either half is independently checksummed, reloadable, and usable if the
   other job fails; and
3. the data is large enough for an initial controlled fine-tuning experiment
   while remaining cheap to replace after a diagnosed mistake.

## Gates before training

Both jobs must pass the full test suite before generation. Afterward, reload
every chunk and check its checksum, terminal outcome, seed range, visit count,
model digest, symmetry setting, and search configuration. Report diversity,
seat balance, length, policy entropy, and target disagreement for each shard
and for the combined set.

Do not start a long self-play loop yet. First fine-tune from the selected
pretrained checkpoint on this fixed data, compare value targets on identical
game-level splits, and evaluate agents under equal wall-clock move budgets.
If the trained agent regresses, diagnose policy legality/orientation, replay
mix, optimization, and evaluator/search interaction before generating more
games.
