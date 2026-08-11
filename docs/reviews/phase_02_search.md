# Phase 2 review: PUCT before neural networks

## Algorithm at the start of this phase

The rules, relative policy mapping, and four exact symmetries pass their
reference tests. Search receives absolute `Move` objects and an evaluator that
returns 192 policy weights plus one **absolute Player 1 value**.

A simulation descends through PUCT, evaluates one new leaf, and backs that same
absolute value up to every node on the path. There is no alternating negation.
At a Player 1 node selection maximizes `Q + U`; at a Player 2 node it maximizes
`-Q + U`, which is equivalent to minimizing absolute Q. An unvisited child's Q
is exactly its parent's current Q.

The rules change turns after an ordinary move but not after the terminal move.
Search therefore copies `to_move` from the resulting `GameState`; it never
assumes that every child has the opposite player. Terminal nodes are backed up
directly from their absolute winner and are never sent to an evaluator.

## Dummy evaluator

The first evaluator gives every policy slot equal weight. Search, not the
evaluator, masks illegal actions and normalizes the remaining weights. Its
value is the result of a uniformly random rollout. This deliberately separates
MCTS correctness from Keras correctness.

## Tests required before optimization

1. Root state remains byte-for-byte unchanged.
2. Illegal actions never become children, even if they receive all raw prior.
3. Legal priors normalize to one, with a uniform fallback after a zero mask.
4. Root and child visit accounting is exact.
5. Backup stores the same absolute result at Player 1 and Player 2 nodes.
6. Player 1 prefers larger Q and Player 2 prefers smaller Q.
7. An unvisited child reads its parent's Q.
8. Search finds a one-ply win for either color.
9. Fixed seeds give fixed rollout and search results.
10. A terminal child keeps the last mover for either color and is never
    mistaken for a non-terminal node merely because turns normally alternate.

## Likely bottlenecks

During dummy pretraining, random rollouts dominate rather than move generation.
During neural self-play, small unbatched CNN calls will dominate. The first
implementation therefore favors a clear evaluator interface. We will profile
on a short HPC pilot before considering tree reuse, parallel actors, or batched
leaf evaluation.

Root Dirichlet noise is supported but disabled in deterministic tests. Playout
cap randomization belongs in the self-play driver, not in PUCT itself.

## What is intentionally absent

There is no transposition table, virtual loss, Gumbel search, forced playout
pruning, or dynamic PUCT scaling. Each can obscure the invariants above and
none is needed to validate the baseline.
