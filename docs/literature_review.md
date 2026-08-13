# Literature and implementation survey: compute-limited AlphaZero

Reviewed 2026-08-11. This is a decision document, not a bibliography. A paper
can justify an experiment; it cannot make a constant from Go, Hex, or Othello
correct for Breakthrough. Every adopted idea still needs an equal-wall-clock
5x5 test and a smaller 8x8 confirmation.

## Executive decision

The next coherent baseline is deliberately modest:

1. Use the native 5x5 CNN and compact 75-action policy to establish search,
   architecture, value-target, exploration, and training starting points.
2. Keep parent-Q first-play urgency as a project invariant. Tune simulations
   and `c_puct`; do not silently replace the requested FPU rule.
3. Compare outcome, root search value, and one simple mixture on the same raw
   games. Preserve the sequence and search statistics for later short-horizon
   tests.
4. Test fixed search against KataGo-style playout-cap randomization. Test
   ordinary PUCT against a faithful Gumbel implementation only after the PUCT
   baseline is stable.
5. Batch independent games on the GPU and parallelize immutable data shards on
   the cluster. Do not rewrite the teaching project in JAX or CUDA merely to
   reproduce a throughput paper.
6. Use a bounded recent replay window, but select its capacity only after
   measuring positions/hour and learner reuse. Always retain the raw archive.
7. Promote settings from 5x5 as informed priors, not truths. Confirm locally
   on 8x8 before scaling data.

## Evidence matrix

“Regime” matters: a result from thousands of accelerators, 32,000 parallel Hex
boards, or a 19x19 Go engine is not a Breakthrough default.

| Source / technique | Mechanism and claimed benefit | Evidence and compute regime | Complexity here | Decision |
| --- | --- | --- | --- | --- |
| [OLIVAW](https://arxiv.org/abs/2103.17228): heavily visited off-trajectory states | Add searched-but-unplayed states, their visit policy, and root `Q`; approximately double labels per self-play game | Othello; about 50,000 games, 10-block CNN, 100/200/400 simulations during training, commodity/free cloud compute; no controlled ablation of every choice | Medium: interior records are correlated and need exact state/policy validation | **Test after baseline.** First measure whether extra labels improve fresh-game Elo per self-play hour. Never mix them invisibly with on-trajectory labels. |
| OLIVAW: increasing search with generation | Spend 100, then 200, then 400 simulations as the network improves | Successful single Othello run, not a causal ablation | Low | **Adopt staged budgets**, but advance only when learning curves justify it. |
| OLIVAW: resignation | End very confident games; save moves | Their later human match exposed late-game weakness and they resumed playing every game to the end | Low code, high bias risk | **Reject for now.** Breakthrough games are cheap enough to finish and terminal labels are valuable. |
| [KataGo](https://arxiv.org/abs/1902.10565): playout-cap randomization | Full search on a minority of moves; cheap search elsewhere; only reliable policy searches get normal policy weight | Controlled Go ablations; largest reported general efficiency gain among its tested methods; still a many-GPU Go regime | Low–medium | **Test first.** Compare with fixed PUCT at equal self-play wall time, positions, and downstream training time. |
| KataGo: global pooling | Add global summaries to spatial features | Controlled Go ablation improved sample efficiency | Low | **Test** as one CNN architecture variant after native baseline. Breakthrough has global race/material information, so transfer is plausible. |
| [KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md): auxiliary soft policy | A second head predicts a temperature-softened search policy, enriching gradients on secondary moves | Reported useful in KataGo's later large Go runs; coupled to policy-target pruning | Medium | **Test later**, only if baseline policy calibration shows weak secondary-move learning. |
| KataGo: forced playouts and policy-target pruning | Force exploration, then remove visits caused only by the forcing from the target | Controlled Go ablation showed a smaller gain | Medium–high and easy to implement subtly wrong | **Park.** Revisit only after the simpler exploration tests. |
| KataGo: shaped Dirichlet noise | Concentrate some noise on low-prior but not absurd moves | KataGo explicitly calls its evidence suggestive, not controlled | Medium | **Reject as a baseline.** Ordinary total-concentration noise remains a small 5x5 ablation. |
| KataGo: uncertainty, dynamic `c_puct`, optimistic policy | Extra heads alter search weighting or target selection | Later engine gains, but several interacting features and Go-specific score signals | High | **Park.** They fail the teaching-complexity test today. |
| [Gumbel AlphaZero](https://openreview.net/pdf?id=bERaNdoegnO) and [DeepMind Mctx](https://github.com/google-deepmind/mctx) | Sample root actions without replacement; sequential halving plus completed-Q transformation produces an improved policy target | ICLR 2022; improved Go/chess/Atari particularly at few simulations. Mctx's reference implementation considers at most 16 root actions by default | High but unusually relevant because our neural searches are tiny | **Test as the main search alternative.** Implement the complete algorithm in an isolated module; a root-only Gumbel trick must not be mislabeled “Gumbel AlphaZero.” |
| [Scaling Scaling Laws with Board Games](https://arxiv.org/pdf/2104.03113) | Small networks, fully accelerator-vectorized games/search, regularized search, large learner batches, recent buffer | Hex; up to 32k GPU-parallel boards. The authors report `c_puct` near 1/16 in their setup and explicitly say the changes were not fully ablated | Low for small networks/batching; enormous for a GPU rules rewrite | **Adopt the questions, not the constants.** Sweep small CNNs, batch size, `c_puct`, and replay age. Keep the current batched Python/Keras boundary. |
| [Pgx](https://github.com/sotetsuk/pgx) and [Mctx](https://github.com/google-deepmind/mctx) | JIT/vectorize many environments and searches on accelerators | Pgx reports 10–100x simulator speedups and demonstrates Gumbel AlphaZero; JAX ecosystem, large batches | A full framework rewrite conflicts with Keras and teaching clarity | **Borrow reference tests and batching ideas; reject a rewrite** unless profiling later proves Python rules/search dominate total project time. |
| [Train on Small, Play the Large](https://arxiv.org/abs/2107.08387) | A size-independent GNN, trained across small board sizes, transfers weights to larger graphs; subgraph sampling reduces uncertainty | Othello/Gomoku/Go, one Titan X plus CPU per run, five-run errors; reported strong large-board transfer | High: replaces the required Keras CNN and its policy boundary | **Use as support for the mini curriculum, not weight transfer.** Our native 5x5 and 8x8 CNNs remain separate. A GNN is a later research branch, not a reason to pad 5x5. |
| [Go-Exploit](https://arxiv.org/abs/2302.12359) | Start some trajectories from an archive of interesting states, reaching deeper states and producing more independent value targets | Connect Four and 9x9 Go; 30 validation runs; four A100 GPUs per experiment | Medium; our raw trajectory archive already contains candidate states | **Test later on 5x5** after ordinary self-play is stable. It is a principled alternative/complement to indiscriminate action noise. |
| [PCZero](https://proceedings.mlr.press/v162/zhao22h.html) | Add path-consistency constraints using trajectories and scouted search paths | Hex, Othello, Gomoku; 900k self-play games | High and changes the loss/search contract | **Park.** Interesting, but too much machinery before the simpler target tests. |
| [Value targets in off-policy AlphaZero](https://ir.cwi.nl/pub/30870/30870.pdf) | Replace or combine final result with values derived from the search | Direct experiments include 6x6 Breakthrough; search-derived targets learned faster there | Low; the raw schema already stores the required root/action values | **Test now** on fixed 5x5 data, then confirm the winner on 8x8. |
| [KataGo short-term targets](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#short-term-value-and-score-targets) | Auxiliary heads predict a discounted future search assessment rather than only the final result | Later KataGo runs; intertwined with score and uncertainty heads | Medium | **Test only after** outcome/root-Q/mixture. Our sequential raw records preserve enough information to derive a simple future-root-Q target without new self-play. |
| [KataGo analysis notes](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md) on symmetry averaging | Average more transformed evaluations to reduce root noise | Engine default is one symmetry; more may improve quality but costs inference | Already implemented | **Test at equal move time.** Four-way averaging is not a correctness requirement and is not the default merely because it helped a pilot. |
| [Targeted-search practitioner report](https://www.reddit.com/r/MachineLearning/comments/1tvw6sc/analysis_of_alphazero_training_data_d/) | Reports later models beating predecessors while failing to improve against MCTS/greedy anchors | One 2026 Othello project; uncontrolled forum evidence | None | **Diagnostic lead only.** It mirrors the warned failure mode and reinforces fixed anchors, value calibration, and replay audits; it proves no hyperparameter. |

No public OLIVAW implementation was found during this survey. Its paper is
therefore evidence for a candidate mechanism, not a code reference we can copy
or independently verify line by line.

## Policy and coordinates

AlphaZero's chess policy is spatial: a source square plus move-type planes,
oriented to the player to move. Breakthrough needs only three directions. The
standard CNN therefore emits `8 * 8 * 3 = 192` logits; the native mini CNN
emits `5 * 5 * 3 = 75`. The rules and stored moves remain absolute 8-stride
coordinates. Only `policy_index()` and `decode_policy_index()` cross into the
compact mover-relative policy.

Player 2 positions and moves rotate 180 degrees at the network boundary. A
third input plane identifies whether the mover is absolute Player 1. This
allows the value head to predict Player 1's value directly. Values are never
negated in backup or at inference.

Sources: [AlphaZero chess representation](https://arxiv.org/pdf/1712.01815),
[Leela Chess Zero encoder](https://github.com/LeelaChessZero/lc0/blob/master/src/neural/encoder.cc).

## Value-target experiment

Run these on the same immutable games, game-level split, initialization seeds,
optimizer examples, and evaluation openings:

| Candidate | Formula, always from Player 1's point of view | Purpose |
| --- | --- | --- |
| Outcome | `z` | Unbiased terminal anchor, but high variance and policy-dependent |
| Soft-Z | root mean `q` | Low-variance search assessment; bootstrapped once leaves are neural |
| Mixture | `0.5 * z + 0.5 * q` | Simple bias/variance compromise; coefficient is frozen for the first test |

The first comparison is pretraining with rollout leaves, where `q` is not
bootstrapped from the same network. In neural replay, keep a nonzero outcome
component or a separate outcome anchor until calibration proves otherwise.
Later, test one auxiliary short-horizon head derived from a future root `q`;
do not add several KataGo heads together and make attribution impossible.

## Exploration and search experiment

The 5x5 sweep is broad but deliberately finite:

- Fixed PUCT simulations: `8, 16, 32, 64`.
- `c_puct`: include a low, middle, and high value selected after a diagnostic
  scale check. Do not import `1/16`, `1.5`, or `2.0` as truth.
- Parent-Q FPU: fixed by project requirement.
- Root exploration: no Dirichlet; moderate total-concentration Dirichlet; and
  no Dirichlet plus saved/archive starts. Change temperature separately.
- Search allocation: fixed search versus one playout-cap schedule at equal
  total wall time.
- Symmetry evaluation: one versus four at equal move time.
- Tree persistence: no reuse versus checked child-subtree reuse, after a
  microbenchmark measures the saved evaluations and the bookkeeping cost.
- Gumbel: only the surviving low-simulation budget, compared at equal time and
  with its own correct improved-policy target.

Tiny runs may reject broken or grossly slow settings. They do not select a
winner. A claim needs fresh paired openings, color reversal, uncertainty, and
at least one confirmation seed.

## Architecture experiment

Use only native 5x5 inputs/outputs. Start with `32x3` and `64x4` residual CNNs,
then one smaller candidate suggested by measured inference/learning curves.
Test global pooling as the only structural variant in the first round. The
policy and value heads remain plain and independently inspectable.

The GNN paper proves that cross-size weight transfer is possible with a
scale-compatible architecture. It does **not** justify padded CNN inputs or
claim that ordinary 5x5 CNN weights can be transplanted into an 8x8 CNN. This
project transfers experimental priors first. A GNN branch is worthwhile only
after the CNN baseline is trustworthy and if the research question justifies
its extra educational load.

## Replay and systems decisions

All raw chunks remain immutable. The online replay buffer is only a FIFO view
over that archive. Choose short/medium windows in units of newly generated
positions, and report:

- positions and complete games in the window;
- age by checkpoint and wall time;
- optimizer examples per new position;
- source-model fractions;
- throughput, policy surprise, and value calibration by age;
- Elo against pure MCTS, pretrained PUCT, alpha-beta, and older neural anchors.

Do not place an accept/reject arena in the actor path. AlphaZero explicitly
contrasts its continual single-network updates with AlphaGo Zero's 55%
best-player gate: subsequent self-play uses the latest parameters and omits
the evaluation/selection step. We follow that simpler control flow. Evaluation
still runs on immutable paired openings, but only as a diagnostic learning
curve and regression alarm.

Batch neural inference across independent games. Shard generation through
Slurm arrays with disjoint deterministic seeds, then validate manifests and
deduplicate seeds before training. Actor count is increased only while GPU
utilization and positions/hour improve; otherwise more parallelism merely
creates stale data and scheduler overhead.

Sources: [AlphaZero algorithm description](https://arxiv.org/abs/1712.01815),
[OpenSpiel AlphaZero notes](https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md),
[Leela training-data formats](https://lczero.org/dev/wiki/training-data-format-versions/),
[replay-buffer design](design_replay_buffer.md).

## Ranked implementation queue

1. **Implemented:** native 5x5/75 network, rules-derived ply bound, balanced
   four-symmetry augmentation, immutable data, and reproducible run metadata.
2. **Now:** smoke-test the continuous bounded-replay learner, then measure a
   short 5x5 learning curve using the latest actor at every update.
3. **Then:** 5x5 search/allocation/exploration tests, including playout-cap
   randomization and equal-time symmetry averaging.
4. **Then:** transfer the coherent winners as 8x8 starting values and perform
   a narrow local confirmation.
5. **Only after stable learning:** Gumbel search, archive starts, OLIVAW-style
   interior labels, global pooling, and one short-horizon auxiliary head as
   individually attributable experiments.
6. **Not now:** shaped noise, uncertainty-weighted search, optimistic policy,
   PCZero, a JAX/CUDA rewrite, or a GNN replacement.
