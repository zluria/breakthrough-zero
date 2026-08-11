# Architecture review 01: bird's-eye view

## Overall assessment

The project has the right spine for a research teaching codebase: one absolute
rules representation, a literal rules oracle beside the optimized bitboards,
an evaluator boundary between search and models, architecture-independent raw
games, and invariant-heavy tests. The code is still small enough that a student
can trace one move from rules to policy index to search statistics.

The main architectural risk is no longer rules correctness. It is allowing the
simple scalar MCTS to dictate an unbatchable neural self-play system. The
changes below should happen before the first neural HPC run, not as patches
after GPU utilization is found to be poor.

## Boundaries to keep

- `game.py` owns rules, absolute moves, the one policy conversion, and the
  three-plane network encoding.
- `symmetry.py` owns the four exact transformations.
- `search.py` knows PUCT and absolute value statistics but nothing about Keras.
- `evaluators.py` supplies replaceable leaf evaluators.
- `data.py` stores absolute states/search statistics and does not import Keras.
- Training, self-play orchestration, arena, and agents will be separate modules.

The game module containing its small encoding is acceptable here: it keeps the
orientation conversion next to the action conversion and makes their shared
invariant visible.

## Changes required before neural self-play

### Incremental and batched search

`PUCTSearch.run()` is intentionally scalar. Keras must not be called once for
each leaf. Refactor the simulation into three operations--select a leaf, batch
evaluate non-terminal leaves from many active games, then expand/backup. The
scalar implementation remains the executable reference. The batched path must
share selection, expansion, and backup helpers rather than copy their logic.

### Lazy state caching

A local Breakthrough benchmark found make/unmake 10--18% slower than
clone-and-replay: unmaking doubles the bitboard updates, while a state clone is
small. Replaying a depth-16 path cost about 40 microseconds. Creating one lazy
child state cost about 3 microseconds and a cached state about 220 bytes.
Controlled end-to-end PUCT A/B tests refined the decision: replay was 10% faster
at 32 simulations, while lazy caching was 7--8% faster at 100 and 400. Search
therefore uses a hybrid for the intended full-search regime: the root and
visited nodes cache read-only states; unvisited children do not. A second replay
path for cheap searches is deferred until its gain is material beside evaluator
cost. This also supports tree reuse without putting copies on every legal edge.

The same incremental operation enables a wall-clock search deadline. Fixed
simulation counts are useful diagnostics, but arena agents require untimed
warm-up followed by equal real time per move.

### Preserve network prior separately from search prior

Dirichlet noise changes the prior used by PUCT. If it overwrites the network
prior, later policy-surprise, calibration, or noise experiments cannot be
reconstructed. Nodes and raw chunks therefore retain both. The search prior is
used for selection; the network prior is immutable evidence.

### Configuration and provenance

Every command will consume a serializable configuration and produce a run ID,
resolved configuration, RNG seeds, code revision, environment information, and
wall-clock counters. There should be one configuration path, not constants
repeated in scripts. HPC job-array index may choose a seed/chunk but must not
silently alter an algorithm.

### Failure-safe chunks

Self-play workers write small independent chunks. A chunk should be written to
a temporary name, checksummed, and renamed only when its manifest is complete.
Interrupted workers then waste at most one chunk. Aggregation reads manifests;
it does not rewrite all games into one fragile file.

## Planned Keras model

The first model should be deliberately ordinary:

- Input: `8 x 8 x 3` (ours, theirs, mover-is-absolute-Player-1).
- A small 3x3 convolutional stem and a short residual tower.
- Policy: a 1x1 convolution producing three logits at every source square.
- Value: a small convolution, flatten/dense layer, and one `tanh` scalar.

The value scalar is trained as absolute Player 1 value directly. There is no
player sign layer. Legal masking remains in search, never in the model.

Channel width and number of residual blocks are experiment parameters. The
fixed MCTS pretraining set is used to compare a small grid under equal training
wall time. Global pooling and an opponent-next-policy auxiliary head are
separate later ablations. They do not enter the baseline architecture.

## Self-play and training shape

CPU pretraining workers are embarrassingly parallel: one deterministic seed
range and one output chunk per HPC array task. Neural self-play instead runs a
moderate batch of active games per GPU worker so leaf evaluations are dense.
The learner consumes immutable chunks through a bounded replay window and
records data age; adding actors indefinitely would create stale experience.

The expensive online loop begins only after the fixed pretraining dataset has
selected a sensible value target, network size, optimizer schedule, and search
constants. The experiment is wall-clock constrained end to end, not iteration
constrained.

## Baselines and arena

The alpha-beta baseline will use iterative deepening and a short, explicit
absolute evaluation (material plus advancement). It maximizes for Player 1 and
minimizes for Player 2. The arena controls clocks outside agents, warms each
agent once, pairs every opening with colors reversed, and saves every game.

Elo is a summary with confidence intervals, not a promotion oracle. Direct
paired results remain the evidence for a code or research change.

## Complexity budget

No feature enters the main loop merely because a modern engine uses it. A
variant needs a named hypothesis, isolated configuration switch, equal-time
comparison, and a conclusion entry. Techniques that demand duplicated search
logic or several coupled heuristics remain parked. This is the main defense
against a pile of patches.
