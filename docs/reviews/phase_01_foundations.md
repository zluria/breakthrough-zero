# Phase 1 review: rules and representations

## Decision

The game and search use absolute coordinates. Player 1 always moves toward
larger row numbers, Player 2 toward smaller row numbers, and every outcome is
from Player 1's point of view.

Only the neural-network boundary uses current-player coordinates. A Player 2
position or move is rotated 180 degrees before it reaches the network. The
policy is an `8 x 8 x 3` tensor: source square followed by forward-left,
forward, or forward-right relative to the player moving.

The input has mover-oriented `ours` and `theirs` planes plus a constant
`mover-is-Player-1` plane. Without that third plane, player swapping would make
the network input identical while the required absolute value changed sign.
The value head therefore learns an absolute target directly; there is no
relative value inside the model.

## Why this is the simplest safe design

- `Move` never changes meaning inside the rules, search, or user interface.
- One pair of functions owns the absolute/canonical conversion.
- Both players share the same three policy move types.
- Illegal output slots are expected and are masked after inference.
- A slow, literal move generator acts as an executable specification for the
  optimized bitboard generator.

## The four augmentations

There are two independent symmetries and therefore four combinations:

1. Identity.
2. Reflect left and right.
3. Swap players and reflect top and bottom.
4. Apply both operations.

Player swapping must include top-bottom reflection. A label swap alone is not
a symmetry because it changes the direction in which each piece may move.

Canonicalization turns the piece layouts into only two distinct spatial
patterns. The absolute-player plane keeps all four full inputs distinct. We
retain every transformation so policy mappings and absolute value signs can be
tested directly.

## Main risks before the next phase

1. A shift crossing the left or right edge in the optimized move generator.
2. A Player 2 move being rotated for encoding but not decoding.
3. Transforming a state without applying the same symmetry to its policy.
4. Negating values during MCTS backup despite the absolute-value invariant.

The first three are covered in this phase. The fourth will be addressed with a
small hand-calculated PUCT tree before adding the neural network.
