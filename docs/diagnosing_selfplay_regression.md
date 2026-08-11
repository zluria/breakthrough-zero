# Diagnosing the "self-play made it worse" failure

The project treats this ordering as an expected alarm:

`rollout MCTS > pretrained PUCT > self-play PUCT`

It must be reported honestly. The latest checkpoint never overwrites the
pretrained anchor or raw games.

## Regression ladder

At fixed wall-clock move limits, every online checkpoint plays paired openings
against:

1. Pure rollout MCTS.
2. The immutable pretrained PUCT checkpoint.
3. The previous online checkpoint.
4. The alpha-beta baseline.
5. A small pool of older neural anchors.

Uniform-random openings are fixed before the match and color-swapped in pairs.
The mini game uses two moves per side. Rated search is noise-free after the
opening.

## Isolate the failing component

If full PUCT regresses, evaluate four combinations using the same search and
clock:

| Policy | Leaf value | What it tests |
| --- | --- | --- |
| Uniform | Rollout | Search/rules anchor |
| Network | Rollout | Policy head alone |
| Uniform | Network | Value head alone |
| Network | Network | Full learned agent |

This factorial check is much more informative than retraining immediately.
Also inspect fixed pretraining validation, recent validation, policy entropy,
illegal probability mass, value calibration by game phase, and direct examples
of policy encode/decode.

## Common causes to check in order

1. Absolute-value sign or Player 2 PUCT direction.
2. Terminal leaves evaluated as non-terminal, or turn changed after the final
   move.
3. Policy orientation, illegal mask, or symmetry mismatch.
4. Training on played one-hot moves instead of root visit distributions.
5. Logit/probability or loss configuration mismatch.
6. Batch-normalization training/inference mismatch.
7. Excessive optimizer reuse, learning rate, or a replay window that is too
   small and correlated.
8. A window so large that obsolete weak targets dominate.
9. Too little exploration and policy collapse, or so much noise that small
   searches cannot recover tactical moves.
10. Too few simulations, making bootstrapped neural targets worse than terminal
    rollouts.
11. A search/value target feedback loop such as poorly calibrated soft-Z.
12. Unequal evaluation compute or repeated identical games masquerading as an
    Elo estimate.

The repair must match the diagnosis. We do not respond to every regression by
adding heuristics.
