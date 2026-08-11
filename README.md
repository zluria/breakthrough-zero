# Breakthrough Zero

This is a teaching project: a compact AlphaZero-style agent for Breakthrough.
Correctness and understandable Python come first. An optimization stays only
when a simple reference implementation or invariant test can check it.

For a standalone student guide, see [Teacher's Tips for Designing and Training
Your AlphaZero Agent](docs/teachers_tips_alphazero.pdf) and its editable
[Markdown source](docs/teachers_tips.md).

## Rules used here

The target game is `8 x 8`. Player 1 begins on rows 0 and 1 and moves toward
row 7; Player 2 begins on rows 6 and 7 and moves toward row 0.

The project also has a `5 x 5`, one-starting-row variant as a debug ladder.
It is not a separate toy implementation: both variants use the same rules,
symmetries, search, and data pipeline. The rules use an 8-stride bitboard
internally, but that implementation detail stops at the neural boundary. The
mini CNN receives a native `5 x 5 x 3` input and emits 75 policy logits; the
standard CNN receives `8 x 8 x 3` and emits 192. The model loader rejects a
checkpoint from the other ruleset before inference.

We use 5x5 as a real, inexpensive tuning sandbox, not merely a plumbing test.
Its results provide starting values for 8x8, followed by narrow confirmation.
The CNN weights remain separate: no padding, resizing, or cross-size weight
transfer is implied.

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
square representation used by both rulesets and search. Padding squares are
never legal in the mini game, and tests compare both optimized generators with
the literal rules oracle through complete random games.

## Policy representation

The policy head has shape `side x side x 3`: 75 logits on 5x5 and 192 on 8x8.
A policy index selects a compact active-board source square in the current
player's view and forward-left, forward, or forward-right. There are no mini
padding logits.

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

The first value-target comparison uses final result, absolute root MCTS value
(soft-Z), and their fixed half-and-half mixture on identical raw games. Direct
experiments on 6x6 Breakthrough found that search-derived targets learned
faster, but this is evidence for a comparison rather than permission to crown
soft-Z in advance. The raw data retains final results and detailed search
statistics so targets can change without regenerating games.

Training applies one exact augmentation per position per epoch using a
balanced four-epoch cycle. Validation uses identity examples. This gives every
stored position all four symmetries without storing four correlated copies.

See the expanded [literature and implementation
survey](docs/literature_review.md) for OLIVAW, KataGo, Gumbel AlphaZero,
resource-efficient implementations, cross-size GNN work, value targets, and
the explicit adopt/test/park decisions.

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

Rated games start from immutable, candidate-independent opening suites. On the
mini board, a uniform random agent makes exactly two moves per side; rated
search is then completely noise-free. Every opening is used by every matchup
and played twice with colors reversed. Early finite-sample Elo uses one
explicit virtual drawn pair, and random remains the fixed 1000-point anchor.

The common case where self-play makes the learned agent weaker has an explicit
[regression ladder and diagnostic procedure](docs/diagnosing_selfplay_regression.md).

## Correctness strategy

The optimized legal-move generator uses Python integers as 64-bit bitboards.
Beside it, `reference.py` contains a slow generator that scans every square and
states the rules literally. Tests compare them through many random games.

PUCT caches a state only after a node is visited. This measured hybrid was
faster than replaying moves from the root and avoids storing states for
unvisited legal children; make/unmake was slower for this game's tiny state.

The tests also check policy round trips, make/unmake, every symmetry, absolute
outcome signs, terminal-node handling, and the rules-derived safe game bounds
(40 mini plies, 208 standard plies). Hand-calculated trees test absolute PUCT
backup, selection direction, legal masking, and parent-Q initialization.

Each major phase begins with a review in `docs/reviews/` covering the current
algorithm, correctness risks, bottlenecks, and the smallest justified next
step.

## Development setup

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run the smallest end-to-end self-play check with:

```bash
python scripts/generate_dummy_selfplay.py work/pilot \
  --rules mini --games 2 --chunk-games 1 --simulations 8 --seed 20260811
```

The generator writes immutable NPZ chunks and publishes a checksummed JSON
manifest last. Re-running the same command verifies complete chunks and skips
them. A configuration, index range, seed, or checksum mismatch fails loudly.

The rules, dummy-evaluator PUCT, reusable raw-data schema, deterministic
self-play generator, wall-clock alpha-beta baseline, Keras CNN, and fixed-data
learner are implemented. The old 896-game mini screen used a padded network;
its raw games remain useful but its architecture/Elo conclusions are archived
as pipeline evidence only. The old 8x8 soft-Z checkpoint is named
`bootstrap-v0`, not “the winner.” Small neural pilots established that batching
works; they did not establish that four-way symmetry averaging, no noise, or a
particular simulation count is optimal.

The current gate is the native 5x5/75 TensorFlow test on an RTX 3070, followed
by fixed-data target/architecture comparisons and literature-guided equal-time
5x5 search tests. See [project status](docs/project_status.md), [HPC
operations](docs/hpc.md), the [external audit](docs/reviews/external_audit_20260811.md),
and the [full adversarial audit](docs/reviews/adversarial_project_audit_20260811.md).
