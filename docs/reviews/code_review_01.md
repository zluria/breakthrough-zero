# Seasoned code review 01: before any large run

This review treats every convenient assumption as suspicious. A fast wrong
self-play generator would be the most expensive failure in the project.

## Findings already fixed

### Terminal turn handling (critical)

The first version changed turns even after the winning move. That convention
is especially dangerous in an alternating game because a terminal leaf can
look like a position belonging to the loser. A terminal `GameState` now keeps
the last mover, and `Undo` records that mover instead of inferring it by
negation. Search copies the resulting state's player into a traversed child.

### Absolute value after spatial canonicalization (critical)

Mover-oriented piece planes erase which absolute player is moving. A network
with only those planes could not predict an absolute Player 1 value without a
hidden relative-value conversion. A third constant plane now identifies
Player 1. The value head will learn and return the absolute target directly.

## Search audit

- Terminal positions are checked before evaluator calls.
- Backup adds the same absolute number at every depth.
- Only PUCT selection uses the player: `Q` for Player 1 and `-Q` for Player 2.
- An unvisited child reads its parent's live Q.
- Illegal priors are masked and legal priors are renormalized, with a uniform
  fallback when the legal mass is zero.
- Root expansion counts as the first simulation. Consequently root child
  visits sum to `simulations - 1`; this is tested and must remain documented.
- A non-terminal expansion with zero legal moves now fails loudly. It signals
  a broken terminal invariant rather than producing a division by zero.

Tree reuse, parallel virtual loss, and batched evaluation remain absent. They
are performance changes with their own failure modes and do not belong before
the scalar search is trusted.

## Rules and rollout audit

The bitboard generator has a literal square-scanning oracle and is compared
against it through seeded complete games. File-edge masks are applied to the
source before every diagonal shift. Straight moves use the empty mask;
diagonals use not-ours, so they may move empty or capture but never land on an
own piece.

Random rollout originally built every legal `Move` object just to select one.
The new hot-path sampler counts bits in the same three target masks and creates
only the selected move. Its optional tactical rule is intentionally tiny:

1. Choose uniformly among immediate wins if any exist.
2. Otherwise choose uniformly among captures if any exist.
3. Otherwise choose uniformly among all legal moves.

The neural policy remains uniform. This rollout preference is a leaf-value
heuristic, is configurable, and must be identified in dataset metadata. We
keep it only as an optional experiment if a short benchmark shows a meaningful
cost relative to the fastest uniform sampler. Its playing/target quality is a
later fair ablation, not assumed from speed.

## Data audit

The saved record uses absolute moves and values and retains root/edge first and
second moments. Loading disables pickle, verifies schema and SHA-256, and now
checks game, position, and action counts. Float32 is sufficient for the planned
search budgets, but the manifest must record the maximum simulations; a future
budget above roughly millions of visits would require a precision review.

Entire internal trees are not retained. Greedy-backup value is materialized
while the tree exists. This is the one explicit tradeoff against maximum future
target flexibility.

## Remaining pre-HPC checks

- Run the full unit suite after this review.
- Benchmark fast versus reference legal moves and all rollout selectors.
- Run deterministic, checksum, terminal, and symmetry stress tests with larger
  seed ranges on the HPC CPU queue.
- Profile before optimizing. Expected dummy-search bottleneck: rollout move
  sampling. Expected neural-search bottleneck: unbatched inference.

## State-management benchmark decision

On paths of depth 4--32, resetting and replaying was indistinguishable from a
fresh clone and replay. Make/unmake was 10--18% slower. A lazy cached child cost
roughly 3 microseconds to create versus roughly 40 microseconds to replay a
depth-16 path, at a conservative 220 bytes per cached state. We adopted lazy
states on visited nodes only. A matched complete-PUCT test found replay 10%
faster at 32 simulations but cache 7--8% faster at 100 and 400. We keep cache
for full search and avoid a second path until fast-search profiling justifies
the complexity.
