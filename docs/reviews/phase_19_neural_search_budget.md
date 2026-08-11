# Phase 19 review: choose the first neural self-play search budget

## Evidence entering the phase

The 16-simulation actor was a plumbing gate, not a training recommendation.
The non-ensemble model produced only 9/64 Player-1 wins. Exact four-symmetry
averaging improved this to 18/64, while a learned-policy/zero-value ablation
gave 20/64. Removing the value changed paired outcomes in both directions
(13 toward P1, 11 toward P2), so the value head is not the remaining cause.

A uniform-policy/value-only ablation at 16 simulations is invalid as a head
comparison: standard openings have about 22 legal moves, so the search often
cannot visit every action and deterministic move ordering dominates. Its 0/64
result is retained as evidence that the budget is below that ablation's useful
regime.

## Controlled comparison

Run 32 and 64 simulations with:

- the same selected 64-simulation soft-Z checkpoint;
- exact four-symmetry inference averaging;
- the same 64 game seeds and visit-sampling schedule;
- no root noise;
- `c_puct=1.5`; and
- one RTX 3070 per independent job.

Both runs save immutable schema-3 search records and pass checksum/replay and
diversity audits. Primary decision quantities are end-to-end positions/second,
visit-target entropy/concentration, soft-Z disagreement with final outcomes,
game length, and obvious seat collapse. These are data-quality diagnostics,
not Elo; the selected budget must still survive training and a paired arena.

## Decision rule

Choose 32 unless 64 gives a material target-quality or stability improvement
that justifies its lower generation rate. Do not add root noise during this
comparison: all previous no-noise neural pilots had unique 4-, 8-, and 12-ply
prefixes and more than 95% unique states.
