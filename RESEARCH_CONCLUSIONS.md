# Research conclusions

This file holds general, high-impact findings discovered by the project. The
README explains how the project works; this file explains what experiments
taught us beyond the assignment.

## Evidence standard

Each claimed finding must include:

- The question and one-factor comparison.
- Equal wall-clock budgets and hardware used.
- Training seeds and saved configuration identifiers.
- Paired, color-reversed arena games under equal wall-clock move limits.
- Wins/losses/draws, Elo difference, and a 95% confidence interval.
- Links to raw data, logs, and model checkpoints.
- Scope and important limitations.

Labels used below are **confirmed**, **preliminary**, **negative result**, and
**engineering observation**. Hypotheses belong in the experiment plan, not in
the conclusions.

## Findings

The first playing-strength results are preliminary mini-board baselines. Local
throughput observations still require care when transferred to the HPC:

- **Engineering observation:** bit-sampling one rollout move avoided building
  the full legal list and was about 6.6 times faster in the selector
  microbenchmark. Full uniform rollouts were about 3.5 times faster.
- **Engineering observation:** state strategy depends on search budget. Replay
  was about 10% faster at 32 simulations; lazy visited-node state caching was
  7--8% faster at 100 and 400. The full-search implementation uses lazy cache.
- **Engineering observation:** multi-gigabyte GPU environments should build
  under an unpublished path and write a persistent log independent of the SSH
  observer. One timed-out observer left a harmless 23 MB partial build and a
  reusable 2.9 GB wheel cache; the cached retry passed validation and only then
  atomically published the environment.
- **Engineering observation:** state-management results do not transfer across
  search algorithms. Lazy cached states beat alternatives for full PUCT, while
  one make/unmake path beat cloning by about 20% on mini alpha-beta and 19% on
  standard alpha-beta. The paired benchmark alternated method order because a
  single sequential run gave the wrong 8x8 conclusion.
- **Engineering observation:** evaluation opening depth must scale with game
  horizon. A six-ply noisy prefix on 5x5 already produced an immediate-win
  start. Four uniform-random plies (two moves per side) are simpler, and
  duplicate color reversal cancels nontrivial seat advantage without filtering
  positions using a competing agent.
- **Preliminary:** on 32 duplicate 5x5 opening pairs at 50 ms per move,
  win/capture-preferred rollout PUCT beat plain-rollout PUCT 55-9, about +299
  pair-regularized Elo (95% CI +140 to +459). This contradicted the prediction
  that lower rollout throughput would erase the tactical preference. It is one
  tournament seed and does not yet transfer to 8x8 or a neural agent.
- **Preliminary:** alpha-beta beat plain-rollout PUCT 52-12, about +244 Elo
  (95% CI +99 to +389). Tactical PUCT led alpha-beta 38-26, about +64 Elo, but
  the interval included zero (-54 to +182).
- **Preliminary:** in the valid 896-game neural mini-board screen, soft-Z beat
  final-outcome training 20-12 for both the 32x3 and 64x4 CNNs.  Each comparison
  was about +83 Elo with a wide 95% interval (-81 to +247), so the repeated
  direction is encouraging rather than conclusive.
- **Negative result:** the padded 8x8 tensor and 192-logit head used for the
  5x5 neural smoke test confound mini architecture comparisons. The active
  board has uneven convolutional boundaries and most policy slots are always
  illegal. No mini weights entered an 8x8 checkpoint, but the mini 32x3 versus
  64x4 result must not be used to choose a standard-board architecture.
- **Negative result:** fixed-data pretraining did not yet beat the strongest
  search baselines on the mini board.  The 64x4 soft-Z network lost 9-23 to
  tactical-rollout PUCT and 6-26 to alpha-beta at the same nominal 50 ms move
  budget.  This establishes the pre-self-play regression checkpoint that later
  agents must surpass.
- **Engineering observation:** scheduler grace should classify failures, not
  enlarge an agent's search budget.  With 50 ms passed to every searcher and
  100 ms external grace, all 896 games were valid.  Long-tail rollout calls had
  less search work because the process was descheduled; recording both elapsed
  time and implementation-independent work counts exposed this distinction.
- **Preliminary:** the 64-simulation soft-Z CNN was the strongest neural agent
  in the first 448-game standard-board screen.  It beat its matched outcome
  model 12-4 (+166 Elo, 95% CI -69 to +401), plain-rollout PUCT 10-6, and lost
  7-9 to tactical-rollout PUCT.  Soft-Z was not universally better: on the
  32-simulation data it lost 7-9 to outcome training.  Selection therefore also
  uses soft-Z's substantially better held-out value error.
- **Preliminary:** increasing dummy-MCTS data search from 32 to 64 simulations
  improved the outcome-trained standard agent 16-0 (+492 Elo, 95% CI +92 to
  +893), but improved the soft-Z agent only 9-7.  More expensive targets can
  help, but their value depends on the target and must be judged against the
  measured generation-rate drop.
- **Negative result:** alpha-beta swept the selected standard 64-simulation
  soft-Z agent 16-0 at 50 ms/move.  Batch-one neural PUCT completed only 4.4
  simulations per move, so inference utilization is now a higher-value
  bottleneck than more fixed-data epochs.
- **Engineering observation:** independent-leaf batching transformed GPU
  utilization for the compact CNN. On one RTX 3070, batch 64 processed 4,375
  leaves/s versus 89 at batch 1 (49x) while call latency rose only from 11.3 to
  14.6 ms. This justifies a lockstep pool of independent games before spending
  on neural self-play.
- **Engineering observation:** correct random data augmentation did not make a
  small, underfit CNN approximately symmetric. Across 256 exact pairs, policy
  L1 residuals were 0.23--0.39 and player-swap value residuals were about 0.16.
  Exact four-symmetry inference averaging is cheap enough under batching to be
  a useful bootstrap stability boundary, while retaining absolute-P1 values.
  This does not establish symmetry averaging as an equal-time playing default:
  four evaluations per leaf must still beat one evaluation given the same move
  clock.
- **Engineering observation:** a search budget below the root branching factor
  can invalidate head ablations. With uniform priors, 16 simulations could not
  visit roughly 22 opening moves; legal-move order then dominated and Player 1
  lost all 64 games. This does not show that the learned value head was weak.
- **Diagnostic observation, not a noise conclusion:** every 64-game no-noise
  pilot had unique trajectories, but uniqueness does not show that low-prior
  good moves were reached or that training will avoid a policy blind spot.
  Exploration must be judged by coverage and downstream learning, not a
  duplicate count alone. Noise therefore remains an open ablation alongside
  temperature, Gumbel search, and archive starts.
- **Engineering observation:** compressed replay formats must be audited at
  access boundaries. Repeatedly indexing NPZ members inside the action loop
  made a 2.6 MB reload take about 109 seconds; materializing each array once
  reduced it to 0.87 seconds without changing the schema or data.
- **Engineering observation:** an AI agent's review of its own design is not
  independent assurance. In this project, written phase reviews still failed
  to challenge two conspicuous mistakes: treating a 64-game smoke test as
  meaningful pretraining evidence, and padding the 5x5 experiment into an 8x8
  neural input and policy head. The same agent that chose an approach is prone
  to rationalize it, and a document titled "review" does not remove that bias.
  Students should personally inspect the assumptions, dimensions, scale, and
  experimental controls before approving an expensive run. AI self-review can
  still help when it is adversarial, performed before implementation, and
  backed by executable invariants and deliberately hostile tests; it should be
  treated as an aid to human review, never as its replacement.
- **Experimental-design lesson:** a small-board curriculum transfers evidence
  about mechanisms and useful parameter ranges, but ordinary fixed-size CNN
  weights do not transfer across board sizes. Weight transfer requires a
  genuinely scale-compatible architecture such as a GNN; padding a small board
  merely changes the experiment and can introduce boundary artifacts.
- **Resource-efficiency lesson:** self-play labels can sometimes be increased
  without playing more games by retaining heavily searched off-trajectory
  nodes, as OLIVAW did. Those labels are correlated and bootstrapped, so they
  require an explicit source flag and a fresh-game Elo ablation rather than
  being silently mixed into the baseline.
- **Fairness lesson:** equal epochs are not equal compute across architectures.
  Record wall time and examples processed, keep intermediate checkpoints, and
  evaluate learning curves rather than comparing only whichever final epoch a
  job happened to save.

Raw commands and results are in
[`docs/benchmarks/foundation_hot_paths.md`](docs/benchmarks/foundation_hot_paths.md).
HPC environment evidence is in
[`docs/benchmarks/hpc_smoke_20260811.md`](docs/benchmarks/hpc_smoke_20260811.md).
The depth-first comparison is in
[`docs/benchmarks/alphabeta_state.md`](docs/benchmarks/alphabeta_state.md).
The preserved opening failure and duplicate-opening rerun are in
[`docs/benchmarks/mini_hpc_33478.md`](docs/benchmarks/mini_hpc_33478.md) and
[`docs/benchmarks/mini_hpc_33479.md`](docs/benchmarks/mini_hpc_33479.md).  The
first padded mini neural screen (now retained only as pipeline evidence) is in
[`docs/benchmarks/mini_neural_33516.md`](docs/benchmarks/mini_neural_33516.md),
and the first standard neural screen is in
[`docs/benchmarks/standard_neural_33517.md`](docs/benchmarks/standard_neural_33517.md).
The inference-throughput gate is in
[`docs/benchmarks/network_batching_33518_33521.md`](docs/benchmarks/network_batching_33518_33521.md).
The neural search, symmetry, noise, and sampling pilots are in
[`docs/benchmarks/neural_selfplay_pilots_33522_33531.md`](docs/benchmarks/neural_selfplay_pilots_33522_33531.md).

## Experiment register

| ID | Question | Status | Wall-clock budget | Result |
| --- | --- | --- | --- | --- |
| V001 | Which of outcome, soft-Z, and a 50:50 mixture learns best? | Historical padded result inconclusive; native test planned | Equal training wall time, then fresh paired arena | Old soft-Z interval overlapped zero; repeat natively |
| S001 | Does playout-cap randomization improve Elo per hour? | Planned | Set after HPC pilot | Pending |
| N001 | Does global pooling improve Elo per hour? | Deferred | Not set | Pending |
| N002 | Does an opponent-next-policy auxiliary head improve Elo per hour? | Deferred | Not set | Pending |
| T001 | Which compact native 5x5 CNN is strongest per training hour? | Planned | Equal wall-clock learner budget | Use winner as an 8x8 starting prior, then confirm locally |
| T002 | Which optimizer and schedule are strongest after equal-time training on fixed data? | Planned | Set after HPC pilot | Pending |
| R001 | Do win/capture-preferred rollouts improve Elo per hour over uniform rollouts? | Preliminary mini result | 32 pairs, 50 ms/move | +299 Elo [+140, +459] |
| P001 | Does 64-simulation pretraining data beat 32-simulation data? | Preliminary standard result | 2 x 8 pairs, 50 ms/move | Outcome: 16-0; soft-Z: 9-7 |
| E001 | Which simple exploration scheme improves learning per hour? | Reopened | Equal-time native 5x5 runs | Unique trajectories alone were insufficient evidence |
| B001 | Which replay-window age best balances forgetting and staleness? | Planned | Equal-time short/medium/long windows | Pending |

The full fairness rules are in
[`docs/experiment_protocol.md`](docs/experiment_protocol.md).
