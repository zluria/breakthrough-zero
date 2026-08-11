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

## Result

Jobs 33534 and 33535 completed on separate RTX 3070 nodes in 4:32 and 4:36.
They produced four 64-game chunks: 256 games, 18,110 positions, and about
11.7 MB total. Both shards had 128/128 unique trajectories and unique
4/8/12-ply prefixes, 97.2% unique positions, and 32 root visits throughout.
Player 1 won 45 games in the first shard and 53 in the second (98/256
combined). The difference between shards is a reminder not to tune from one
64-game color statistic.

The original submissions, jobs 33532 and 33533, failed immediately at the
required `MODEL_PATH` check because nested PowerShell/SSH quoting removed a
remote shell variable. They created no output directory or game data. The
replacement submissions used the literal absolute checkpoint path; retaining
the failed IDs makes the operational cause auditable.
