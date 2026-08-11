# Phase 7 review: build a baseline and measure agents fairly

## Baseline design

Alpha-beta uses the same absolute Player 1 values as PUCT. Player 1 maximizes;
Player 2 minimizes. There is no negamax sign conversion. Iterative deepening
keeps the last fully completed depth when the wall-clock deadline interrupts a
deeper search.

The first leaf evaluator has only two explainable terms:

- material difference, normalized by the starting pieces per player;
- advancement difference, normalized by board length and starting material.

Their weighted sum is clipped strictly inside `(-1, +1)`, reserving exact
endpoints for rules-proven terminal results. Goal moves and captures are
searched first but do not add hidden evaluation bonuses.

Depth-first search uses make/unmake because it owns one mutable path. Every
move is reversed in a `finally` block, including deadline exceptions and
terminal moves. This is different from PUCT's lazy immutable node cache and is
the right place to remeasure the state-management choice.

## Deliberate omissions

No transposition table, quiescence search, hand-coded threat detector, or
game-specific opening book belongs in the first baseline. Add one only after a
profile and a named strength-per-second experiment. The baseline's first job
is to expose neural/search regressions, not to become a handcrafted champion.

## Evaluation contract

Agents receive equal wall-clock time per move. The arena records actual time,
move, node/simulation counts where available, seed, code version, and complete
game trajectory. Search noise is off during rated moves.

Deterministic games use a saved opening suite generated before results are
observed. Noise is restricted to the first 5--10 opening plies; each opening is
then played twice with agent colors reversed. This creates diversity without
letting Dirichlet noise contaminate the rated search budget.

Report wins, losses, draws, score, Elo difference, and a confidence interval.
The point estimate never stands alone. A match runner is not complete until it
can reproduce an opening, reject illegal moves, detect time overruns, and prove
that paired assignments are balanced.

## Smallest justified next step

Implement and test the alpha-beta search, including both-color immediate wins,
absolute minimization, heuristic symmetry, timeout restoration, and terminal
turn handling. Then benchmark make/unmake against child cloning on this exact
depth-first workload before building the arena around it.

## Baseline gate result

The alpha-beta tests pass for both colors, absolute values, symmetry, timeout
restoration, and terminal moves. On alternating five-round local benchmarks,
make/unmake was faster than cloning in every paired round: median gains were
20.2% on 5x5 depth-5 positions and 18.7% on 8x8 depth-4 positions. It remains
the depth-first baseline strategy. The next gate is the saved-opening,
paired-color wall-clock arena.
