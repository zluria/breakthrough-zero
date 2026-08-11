# Breakthrough Zero

This is a teaching project: a compact AlphaZero-style agent for Breakthrough.
Correctness and understandable Python come first. An optimization stays only
when a simple reference implementation or invariant test can check it.

For a standalone student guide, see [Teacher's Tips for Designing and Training
Your AlphaZero Agent](docs/teachers_tips_alphazero.pdf) and its editable
[Markdown source](docs/teachers_tips.md).

## Rules used here

The board is `8 x 8`. Player 1 begins on rows 0 and 1 and moves toward row 7;
Player 2 begins on rows 6 and 7 and moves toward row 0.

A piece moves one row forward on every turn:

- Straight forward is legal only when the destination is empty.
- Diagonally forward is legal when the destination is empty or occupied by an
  opponent; an opponent on the destination is captured.
- Reaching the opposite edge wins immediately.
- A player with no legal move loses.

An ordinary move changes the player to move. The terminal move does not: a
finished position records the last mover in `to_move` and the winner in
`outcome`. This explicit invariant prevents terminal leaves from being treated
as positions for the opponent to evaluate.

Rows and columns are zero-based. Square `row * 8 + column` is the one absolute
square representation used by the rules and search.

## Policy representation

The policy head has shape `8 x 8 x 3` (192 logits). A policy index selects a
source square in the current player's view and forward-left, forward, or
forward-right.

Player 1 is already in this view. Player 2 positions and moves are rotated 180
degrees at the neural-network boundary. Thus "forward" has one meaning in the
network although the absolute row deltas are opposite.

The CNN input has three planes: our pieces, their pieces, and a constant plane
that is one exactly when the mover is absolute Player 1. The third plane keeps
absolute player identity after rotation, so the value head predicts Player 1's
value directly.

The game never stores canonical moves. `GameState.policy_index()` and
`GameState.decode_policy_index()` are the only conversion points. Illegal
policy slots are masked before PUCT uses the probabilities.

## Absolute values

Every public value and every MCTS `Q` is from Player 1's point of view:

- `+1`: Player 1 wins.
- `-1`: Player 2 wins.

Values are never negated during tree backup or converted at the network
boundary. The value target and output are absolute. Player 1 maximizes `Q` and
Player 2 minimizes it; this `+Q` versus `-Q` choice in the PUCT formula is the
only turn-dependent value operation. An unvisited child's initial Q is its
parent's Q.

## Four exact symmetries

| Name | Spatial change | Labels | Absolute value |
| --- | --- | --- | --- |
| Identity | None | Unchanged | Unchanged |
| Left-right | Reflect columns | Unchanged | Unchanged |
| Swap players | Reflect rows | Swap P1/P2 | Negated |
| Swap + left-right | Rotate 180 degrees | Swap P1/P2 | Negated |

The row reflection in "swap players" is essential: a label swap alone reverses
the movement rules. Current-player canonicalization gives two distinct piece
layouts, but the absolute-player input plane keeps all four training examples
distinct. This is necessary for an absolute value output.

## Training decisions

Expensive self-play data will be stored as unaugmented absolute positions,
sparse absolute moves and visits, and several value statistics. Policy encoding
and augmentation happen in the loader, so architecture or encoding changes do
not require regenerating games.

The provisional pretraining value target is the absolute root MCTS value
(soft-Z), not only the final result. Direct experiments on 6x6 Breakthrough
found that search-derived targets learned faster. We retain final results and
run a small paired target ablation on fixed pilot data before spending the full
HPC budget.

See the [literature review](docs/literature_review.md) for the evidence and the
explicit adopt/test/park decisions.

Two especially sensitive online choices have dedicated notes: [root
noise](docs/design_noise.md) and the [replay
buffer](docs/design_replay_buffer.md). Noise is opt-in per root, uses total
concentration rather than a blindly copied per-move alpha, and never alters
absolute values.

Research variants are compared at equal wall-clock training and playing
budgets, with paired color-reversed games and uncertainty on Elo estimates.
The [experiment protocol](docs/experiment_protocol.md) defines fairness. New
general findings go in [RESEARCH_CONCLUSIONS.md](RESEARCH_CONCLUSIONS.md), not
in this project description.

The common case where self-play makes the learned agent weaker has an explicit
[regression ladder and diagnostic procedure](docs/diagnosing_selfplay_regression.md).

## Correctness strategy

The optimized legal-move generator uses Python integers as 64-bit bitboards.
Beside it, `reference.py` contains a slow generator that scans every square and
states the rules literally. Tests compare them through many random games.

PUCT caches a state only after a node is visited. This measured hybrid was
faster than replaying moves from the root and avoids storing states for
unvisited legal children; make/unmake was slower for this game's tiny state.

The tests also check policy round trips, make/unmake, every symmetry, and
absolute outcome signs. A hand-calculated tree will test absolute PUCT backup,
selection direction, legal masking, and parent-Q initialization before the
neural network is introduced.

Each major phase begins with a review in `docs/reviews/` covering the current
algorithm, correctness risks, bottlenecks, and the smallest justified next
step.

## Development setup

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The rules and dummy-evaluator PUCT are implemented. The alpha-beta baseline,
Keras model, self-play pipeline, Elo arena, and HPC jobs are added in subsequent
reviewed phases.
