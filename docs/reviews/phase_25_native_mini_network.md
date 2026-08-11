# Phase 25 review: native 5x5 neural boundary

## Decision

The mini game keeps the shared 8x8-stride bitboard internally because that
makes the optimized rules, make/unmake, storage, and symmetries simple. Its
neural boundary will no longer be padded. A mini model receives a `5 x 5 x 3`
tensor and emits `5 x 5 x 3 = 75` policy logits. A standard model continues to
receive `8 x 8 x 3` and emit 192 logits.

The two networks have separate weights and checkpoints. No resizing, transfer,
or mixed-rules training is allowed.

## Small shared abstraction

`Ruleset.active_size` defines the neural board side and policy size. Game
moves remain absolute 8x8-stride square pairs. `GameState.policy_index()` is
the sole conversion to the compact mover-relative policy, and
`decode_policy_index()` is its inverse. `GameState.encode()` writes only the
active square array.

Search validates the policy shape against the current state's ruleset. The
training loader rejects mixed rulesets before making tensors. A Keras evaluator
infers its board side from the saved model and rejects a state from the other
ruleset before inference. These fail-closed boundaries are essential: a 75-logit
mini policy must never be silently interpreted as a 192-logit standard policy.

## Tests before training

Require policy round trips for both players and both sizes; exact symmetry
mapping on both sizes; native tensor and batch shapes; illegal masking;
scalar/batched equivalence; model save/load; and a real mini training batch.
Existing standard checkpoints must retain their 8x8/192 boundary unchanged.

## Implementation status

The dynamic game/search/training boundary and fail-closed model checks are
implemented on branch `audit/native-mini-research-gate`. The local
non-TensorFlow suite passes. A real TensorFlow build/train/save/load check on an
RTX 3070 is still required before this gate is complete.

After that gate, retrain native 5x5 candidates from the preserved raw mini
games. Compare the `32x3` and `64x4` trunks and outcome, soft-Z, and fixed
half-outcome/half-soft-Z targets. New Elo is useful for choosing 5x5 settings
and 8x8 starting priors; weights never cross between board sizes.
