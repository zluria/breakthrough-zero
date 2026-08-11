# Literature review: a compute-limited AlphaZero for Breakthrough

This is a design filter, not a catalogue. The question is which ideas buy
enough playing strength or reliability to justify their cost in a small,
student-readable project.

## Policy and coordinates

AlphaZero's chess policy is spatial: a source square plus a move-type plane,
with the position oriented to the player to move. That suits Breakthrough, but
its action space can be much smaller. We use 64 source squares times three
mover-relative directions: forward-left, forward, and forward-right.

The game state and search remain in absolute coordinates. Only the neural
network boundary rotates a Player 2 position and action by 180 degrees. This
keeps the rules literal and gives the CNN one meaning for "forward".

A third constant input plane identifies whether the mover is absolute Player
1. This lets the value head learn Player 1 value directly despite the spatial
rotation. We never create a relative value and never multiply a value by a
player sign at inference.

Sources:

- [AlphaZero chess and shogi representation](https://arxiv.org/pdf/1712.01815)
- [Leela Chess Zero encoder](https://github.com/LeelaChessZero/lc0/blob/master/src/neural/encoder.cc)

## Value targets

The original AlphaZero target is the final game result. It is a noisy target
and describes the exploratory self-play policy rather than exactly the greedy
search policy used in evaluation.

Willemsen, Baier, and Kaisers directly tested alternatives on Connect Four and
6x6 Breakthrough. Their three search-derived targets learned faster than the
final-result target, while the differences among the three were small on
Breakthrough. The simplest alternative is **soft-Z**: the root MCTS mean value
at the position.

Our provisional choice is therefore:

- Store all targets from Player 1's point of view.
- Use root MCTS value (soft-Z) as the main pretraining target.
- Also store the final result and ingredients for best-child and greedy-backup
  targets.
- Before the expensive run, train final-result and soft-Z models on the same
  fixed pilot data and keep the simpler winner. This is a paired comparison,
  not a large hyperparameter sweep.

Soft-Z becomes partly bootstrapped once a neural network supplies leaf values.
The raw data therefore always retains the final result, and the first
pretraining set uses terminal random rollouts.

Source:

- [Value targets in off-policy AlphaZero: a new greedy backup](https://ir.cwi.nl/pub/30870/30870.pdf)

## Search and data efficiency

KataGo's most relevant low-complexity improvement is playout-cap
randomization: use a full search on a random minority of turns and cheap
searches on the rest. Only full searches provide policy targets. This spends
more of the budget on distinct games and value examples without pretending
that a tiny search gives a high-quality policy label.

We will add it only after fixed-budget PUCT passes the correctness suite. A
pilot will compare a fixed simulation count with roughly 25% full searches and
75% cheap searches at equal total node budget. The fraction and counts are
configuration values, not constants hidden in code.

KataGo also reports benefits from global pooling and an auxiliary prediction of
the opponent's next policy. These are suitable later experiments because each
has a short explanation and a clean ablation. Go-specific ownership and score
heads do not transfer directly. Forced playout pruning, dynamic value scaling,
uncertainty heads, and optimistic policy targets are parked until evidence
justifies their complexity.

Sources:

- [KataGo paper](https://arxiv.org/pdf/1902.10565)
- [KataGo methods notes](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)

## Engineering lessons after AlphaZero

The largest practical improvements are often outside the loss function:

- Batch neural-network inference across games.
- Generate games in parallel, but monitor actor staleness.
- Keep a bounded replay buffer and immutable raw self-play chunks.
- Separate search, data generation, training, and evaluation so each can be
  tested independently.
- Record seeds, search settings, model identity, and root statistics so a bad
  run can be reproduced rather than guessed at.

OpenSpiel documents a clear actor/evaluator/learner layout and warns that too
many actors can make training data stale. Leela's public formats demonstrate
the value of retaining root values, visits, results, and search metadata rather
than saving only already-encoded tensors.

Sources:

- [OpenSpiel AlphaZero implementation notes](https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md)
- [Leela Chess Zero training-data formats](https://lczero.org/dev/wiki/training-data-format-versions/)

## Low-compute evidence

The closest source found to the requested "AlphaZero on a Shoestring" is
*Scaling Scaling Laws with Board Games*. It trained a strong 9x9 Hex agent on a
single RTX 2080 Ti in under three hours. Its transferable lesson is extensive
batching and vectorization with modest searches; its exact Hex constants are
not Breakthrough defaults. We will establish a Breakthrough compute frontier
on the HPC instead of copying them.

Source:

- [Scaling Scaling Laws with Board Games](https://arxiv.org/pdf/2104.03113)

The exact paper title "AlphaZero on a Shoestring" was not located in the
provided files or indexed sources. A link, author, or copy is needed before we
can claim that its recommendations were reviewed.

## Adopt, test, or park

| Decision | Technique | Reason |
| --- | --- | --- |
| Adopt now | Absolute Player 1 values | Required invariant; removes hidden sign changes |
| Adopt now | Relative 64x3 policy | Small, exact, and easy to test |
| Adopt now | Parent-Q first-play urgency | Required and simple |
| Adopt now | Rich immutable raw data | New encoders and targets do not require new games |
| Adopt now | Batched inference interface | Main route to useful GPU utilization |
| Test cheaply | Soft-Z versus final result | Direct positive evidence on Breakthrough |
| Test cheaply | Playout-cap randomization | Strong efficiency evidence and modest complexity |
| Test later | Global pooling; opponent-policy head | Understandable architecture ablations |
| Park | Gumbel search, reanalysis, uncertainty targets | Too much machinery for the first correct baseline |

## Raw data contract

Self-play is stored before augmentation and before conversion to a neural
policy index. Each chunk contains absolute bitboards, player to move, ply,
sparse absolute moves and visit counts, selected move, root visit count,
absolute root value, absolute final result, optional alternative value targets,
full-search/sample-weight flags, and game identifiers. A manifest records the
schema, rules, RNG seeds, MCTS settings, model identity, and code revision.

Augmentation and policy encoding happen in the training loader. Consequently a
future architecture can reuse every game, and an encoding bug can be fixed
without regenerating expensive search data. Training and validation are split
by whole game, never by individual positions.
