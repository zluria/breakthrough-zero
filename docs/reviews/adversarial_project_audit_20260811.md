# Adversarial project audit: 2026-08-11

## Mandate

This is a hostile design review, not a progress summary. It asks whether a
strict supervisor should authorize the next expensive experiment based on the
repository as it exists now. A passing unit test or a document named
"review" is not accepted as evidence that the experimental design is sound.

The audit inspected the README, research conclusions, phase reviews,
benchmarks, experiment protocol, source, tests, Slurm scripts, local Git state,
public GitHub artifacts, and a read-only HPC artifact inventory.

## Executive verdict

**Status: RED for further neural training or generation-to-training loops.**

The rules engine and absolute-value PUCT foundation are credible enough to
continue engineering work. The research pipeline is not yet credible enough
to support expensive training or strong scientific conclusions. The largest
risks are no longer elementary value-sign or terminal-node bugs; they are
experiment scale, mismatch between written protocols and executed jobs,
artifact reproducibility, training/evaluation fairness, and premature
selection from weak evidence.

No Breakthrough-Zero job is currently running. That is the correct state while
these gates are open.

### Evidence that is genuinely good

- The local non-TensorFlow suite passed 83 tests; three TensorFlow tests were
  skipped because the local runtime lacks TensorFlow. The tests explicitly
  exercise absolute backup, Player-1 maximization versus Player-2 minimization,
  parent-Q first-play urgency, terminal moves that retain the last mover,
  policy round trips, legal masking, all four symmetries, make/unmake, and
  scalar-equivalent batched self-play.
- The optimized legal generator is compared with a literal reference over 40
  seeded complete games for each ruleset (`tests/test_game.py:143`).
- Raw records retain absolute states and moves, visits, priors, value sums,
  squared sums, root evaluation, final result, selected move, and alternative
  target ingredients. This is the right architecture-independent boundary.
- Failed and invalid experiments have generally been preserved rather than
  hidden. The job reports distinguish preliminary evidence from confirmed
  results more honestly than many student projects do.
- Independent-game neural batching was tested against scalar search and was a
  real, high-value response to measured GPU under-utilization.

Those strengths do not cancel the findings below.

## Stop-ship findings

### S1. The native 5x5 network does not exist

**Evidence.** The README and implementation still use an 8x8 input and 192
logits for both games (`README.md:19`, `README.md:50`,
`src/breakthrough_zero/network.py:38`,
`src/breakthrough_zero/network.py:79`). Phase 25 describes a future 5x5/75
boundary, not completed code.

**Consequence.** Existing mini neural results are padded-network results. They
cannot select a mini architecture, and no checkpoint currently satisfies the
teacher's request for a separate native mini network.

**Required gate.** Implement ruleset-sized input, policy, search, evaluator,
and training boundaries; make 5x5 and 8x8 checkpoints fail closed when crossed;
pass CPU and real TensorFlow save/load/training tests for both shapes. The old
padded mini checkpoints stay archived and are named `padded-smoke`, never
`mini` without qualification.

### S2. The proposed native-mini training would repeat the 64-game mistake

**Evidence.** Phase 25 proposes retraining from the preserved mini raw games
(`docs/reviews/phase_25_native_mini_network.md:35`). That preserved training
set contains only 64 tactical-rollout games. The project has already concluded
that 64 games are a smoke test, not a meaningful pretraining corpus.

**Consequence.** Fixing tensor dimensions and immediately training on the same
tiny corpus would produce a correctly shaped but still under-evidenced model.
It would repeat the scale error under a new filename.

**Required gate.** It is acceptable to train one clearly labelled native-mini
smoke checkpoint on the old data to prove the boundary. Any rated or selected
mini model needs a larger, staged mini corpus and a nested learning curve. Do
not pick a final game count in advance; inspect at least smoke, intermediate,
and larger prefixes and stop when held-out behavior and Elo cease improving.

### S3. There is no implemented online replay/training loop

**Evidence.** `docs/design_replay_buffer.md` is a design note. There is no
active-window implementation, age accounting, reuse controller, learner/actor
orchestrator, promotion gate, rollback, or generation loop in `src/` or
`scripts/`. The repository currently supports fixed generation, fixed-data
training, and arenas.

**Consequence.** The project is not yet an AlphaZero training system. It is a
well-instrumented collection of its components. Starting repeated manual jobs
now would create an undocumented loop whose replay semantics depend on shell
history and human memory.

**Required gate.** Implement the smallest explicit generation manifest,
position-capped FIFO view, sample-age/reuse accounting, checkpoint identity,
and promotion/rollback state machine before generation 2. The loop must be
restartable and must never overwrite the pretrained or previous-best anchors.

### S4. The first neural update violated the written replay design

**Evidence.** The replay design says the online window is initially seeded
with pretraining games (`docs/design_replay_buffer.md:43`). Job 33536 trained
only on the two new neural shards (`hpc/neural_replay_training.sbatch:25-39`),
with no pretraining archive mixed in.

**Consequence.** This is a sudden distribution replacement from a tiny teacher
corpus to 256 games produced by the weak bootstrap model. It creates a serious
catastrophic-forgetting and feedback-loop risk and is a plausible mechanism
for the known ordering `rollout MCTS > pretrained PUCT > self-play PUCT`.
This is a risk diagnosis, not yet proof: the arena has not run.

**Required gate.** Preserve job 33536 as the zero-anchor-mix experiment. Before
adopting it as the training recipe, run the pending regression arena and, if it
regresses, compare a pre-registered small set of teacher-anchor mixtures on the
same neural data and wall-clock training budget. The implemented replay loop
must match the selected mixture policy.

### S5. "Everything is on GitHub" is false

**Evidence.** The public repository has one release, `neural-replay-r01`, with
one 5.95 MB raw-data archive. It has no checkpoint release and zero GitHub
Actions workflows. The selected standard bootstrap, four standard alternatives,
four padded mini models, and two generation-1 fine-tunes exist only under the
HPC data root. Older pretraining raw data and most scheduler logs are also
HPC-only. Arena metadata records model paths, not hashes
(`scripts/run_mini_tournament.py:118-132`).

**Consequence.** A student cannot reproduce the documented agents from the
GitHub project, and an HPC storage loss would destroy essential evidence.
Paths such as `/home/zurlu/.../model.keras` are not research artifacts.

**Required gate.** Publish versioned releases containing every selected and
comparison checkpoint, `run.json`, source-data manifests or archives, arena
artifacts, and SHA-256 inventory. Add an artifact index to the README. Every
arena must record the model digest, not merely its path.

### S6. Jobs do not prove that they ran committed code

**Evidence.** Slurm scripts use `git diff --quiet` and
`git diff --cached --quiet`. Those checks ignore untracked files. The current
local tree contains untracked Slurm/review files and tracked modifications.
Run metadata records `HEAD`, which would falsely imply reproducibility if an
untracked script or source file affected the job.

**Consequence.** A run can claim commit X while executing code not contained in
commit X. That invalidates the strongest part of the project's audit trail.

**Required gate.** Fail jobs unless `git status --porcelain` is empty; record
the submitted Slurm script's hash; preferably run from an immutable checkout
or exported source bundle. No new job is submitted from an uncommitted phase.

## High-priority findings

### H1. Training comparisons are iteration-matched, not wall-clock-matched

**Evidence.** The protocol requires equal wall-clock training budgets, but
`scripts/train_pretraining.py` stops after epochs. Phase 24 proposes comparing
32x3 and 64x4 networks without a timed training procedure. Existing sweeps gave
architectures the same epochs and examples, not a guaranteed equal time.

**Consequence.** Architecture results can reward the model allowed more FLOPs,
contrary to the teacher's explicit fairness rule. Conversely, a larger model
that is more sample-efficient can be unfairly described using only epoch count.

**Required gate.** Report both equal-example and equal-wall-clock views, with
wall clock primary for playing-strength claims. Pre-register checkpoint times
or optimizer-step snapshots and evaluate the closest completed snapshot under
the same end-to-end budget.

### H2. The learner saves only the last epoch

**Evidence.** Training keeps metrics in memory and saves one model after the
final epoch (`scripts/train_pretraining.py:128-162`). No per-epoch checkpoint
or pre-registered selection rule exists. The generation-1 outcome validation
already worsened after its best observed epoch.

**Consequence.** A later, worse epoch can be sent to self-play, manufacturing
the exact regression the project is trying to diagnose. Training logs cannot
recover discarded weights.

**Required gate.** Save immutable epoch/time snapshots plus the final model.
Choose `best` only by a predeclared metric that is comparable within that
experiment; keep `last` for unbiased reporting. Record output model hashes and
optimizer state policy.

### H3. "Four augmentations" is not guaranteed in short training

**Evidence.** Each sample receives one independently random symmetry per epoch
(`src/breakthrough_zero/training.py:92`). The fixed-data report accurately says
"one randomly chosen" transform, while phase 24 says "all four augmentations."
In a five-epoch fine-tune, most samples will not see all four transforms.

**Consequence.** The absolute-player value balance and symmetry coverage vary
by luck precisely in the short, sensitive self-play update. Exact inference
ensembling then hides rather than fixes learned asymmetry at four times the
inference work.

**Required gate.** Use a simple balanced augmentation schedule that guarantees
the four transforms over a known cycle, or explicitly label and measure random
coverage. Validate on all four transforms and report symmetry residuals; do not
claim all-four exposure when it did not occur.

### H4. The no-noise conclusion tests the wrong failure mode

**Evidence.** The noise design correctly says Dirichlet noise addresses
low-prior blind spots and feedback (`docs/design_noise.md:7-8`,
`docs/design_noise.md:34`). The actual decision used unique trajectories and
positions as the main reason to keep noise off
(`docs/benchmarks/neural_selfplay_pilots_33522_33531.md:59-60,84-85`). The
planned measurements of low-network-prior moves visited were not performed in
the reported table.

**Consequence.** Unique games prove that move sampling creates different
trajectories. They do not prove that a systematically suppressed winning move
is ever explored. Noise may still be unnecessary, but the stated evidence does
not answer its purpose.

**Required gate.** Keep noise off for plumbing, but call the choice
`unresolved`, not a negative result. Before the main loop, run the already
designed fixed-model low-prior exploration diagnostic and only then a small
end-to-end learning ablation if the diagnostic warrants it. Evaluation search
remains noise-free after immutable openings.

### H5. Exact symmetry ensembling was promoted without an equal-time Elo test

**Evidence.** The model had large symmetry residuals, and ensembling improved
self-play seat balance. The review explicitly warned not to promote it silently
because inference has a real cost
(`docs/reviews/phase_18_color_bias_diagnosis.md:34-42`). It was nevertheless
selected from self-play diagnostics, not a paired equal-wall-clock strength
comparison against the raw evaluator. At batch one, neural agents already get
only about four simulations per 50 ms.

**Consequence.** Fourfold inference may make search weaker even while outputs
look cleaner. The effect differs sharply between batched self-play and
batch-one arena play.

**Required gate.** Compare raw versus exact-ensemble agents under equal wall
clock, and separately compare generation throughput and downstream learning.
Treat ensembling as a measured variant, not a correctness requirement.

### H6. The evaluation artifacts do not establish strict equal-time play

**Evidence.** Standard job 33517 gave a nominal 50 ms move budget but allowed
100 ms external grace. Recorded rollout PUCT moves averaged about 53 ms and
had 95th percentiles near 90-94 ms, while neural moves averaged about 46 ms
(`docs/benchmarks/standard_neural_33517.md:57-66`). The arena accepts any move
inside budget plus tolerance (`src/breakthrough_zero/arena.py:261`). Minimum
simulations can also force work that cannot fit inside a very short budget.

**Consequence.** This is not evidence of cheating--the long CPU records often
had low work counts--but it is evidence that supplied budgets and consumed time
are not identical. A strict equal-time claim needs a defined overrun policy and
sensitivity analysis, not the phrase "grace was not available to the agent."

**Required gate.** Pre-register an overrun accounting rule; report actual time
by agent in every conclusion; tune the internal safety margin so means and
tails meet the rule; rerun decisive comparisons at a larger move budget where
one indivisible evaluation is a small fraction of the clock.

### H7. Arena identity and configuration metadata are incomplete

**Evidence.** Tournament metadata contains model paths and ensemble flags but
not model hashes. It does not serialize `c_puct`, minimum simulations,
alpha-beta parameters, or evaluator configuration. Opening suites are
regenerated at runtime from a seed (`scripts/run_mini_tournament.py:134-140`)
rather than loaded from one versioned canonical suite.

**Consequence.** A path can be overwritten, defaults can change, and the same
seed can produce a different suite after move-order code changes. Longitudinal
Elo then ceases to compare the same agents and starts.

**Required gate.** Hash models, serialize every agent parameter, and load a
checked-in or released opening-suite artifact by hash. New opening suites are
separate pre-registered experiments.

### H8. The confidence interval is an undocumented approximation, not a
validated paired-game model

**Evidence.** Pair scores are fractional values in `{0, 0.5, 1}`, then one
virtual draw is added and a binomial Wilson interval is applied
(`src/breakthrough_zero/ratings.py:67-79,176`). Wilson's derivation assumes
Bernoulli observations, not arbitrary fractional cluster scores. The global
table is weighted least squares on regularized pairwise Elo differences.

**Consequence.** The intervals may be conservative in some color-split cases,
but their advertised 95% coverage has not been established. Small reported
effects and promotion decisions cannot lean on them.

**Required gate.** Keep the point estimate descriptive. Add a cluster bootstrap
or a simple Bayesian paired-outcome model, test it on synthetic edge cases, and
report the method. Use head-to-head intervals for claims; keep the fitted pool
table explicitly decorative.

### H9. Fixed-data input compatibility is checked manually, not enforced

**Evidence.** `_load_games` rejects duplicate paths and seeds, and the training
script rejects mixed rulesets. It does not reject different search budgets,
models, noise, sampling schedules, generators, or schema-compatible run
configurations across input roots. Loaded search statistics are checked for
trajectory legality and counts, but not comprehensively for finite priors,
visit accounting, value bounds, or root/action consistency.

**Consequence.** A mistyped directory can silently contaminate a one-factor
experiment while every checksum and unit test passes.

**Required gate.** Define a dataset-compatibility contract. Fixed experiments
fail on mismatched manifests unless an explicit mixture specification names
each component and weight. Validate all numerical invariants at load/audit
time, including `sum(child visits) == root visits - 1` for full searches.

### H10. The 128-ply ceiling is not a rules-derived safe bound

**Evidence.** Self-play and arena defaults cap games at 128 plies
(`src/breakthrough_zero/selfplay.py:33`). Standard Breakthrough can in principle
last longer: a deterministic legal-move stress search during this audit found
a 155-ply standard game. Before reaching a goal, the initial pieces can also
distribute more than 128 total forward advances. Current training games merely
happen to finish sooner.

**Consequence.** A future defensive or pathological network can make valid
games fail generation or become artificial arena draws. This would look like a
training regression while actually being an arbitrary guardrail.

**Required gate.** Derive and document a safe ruleset bound, use it as the hard
bug guard, and separately log long-game percentiles. Do not adjudicate a
theoretically decisive game as a draw at an arbitrary typical length.

## Important research and management debt

### M1. The 2,048-game plan is mislabeled as successive doubling

Phase 24 jumps from 64 to 2,048 games--five doublings--while calling 2,048 the
"first successive-doubling target". The data is cheap and reusable, so 2,048
may ultimately be sensible, but the wording hides the actual commitment.

Submit a small initial subset, audit it, then complete the archive. Train nested
64/128/256/512/1,024/2,048 prefixes from the same valid corpus to obtain the
learning curve that should have chosen the scale in the first place.

### M2. The first neural replay was generated before adequate pretraining

The 256-game neural batch is a useful pipeline diagnostic, but its generator
was the `bootstrap-v0` model trained on only 2,500 positions from 64 games.
It must not be described as the first production generation. Evaluate it for
the known regression, learn from it, and then restart generation numbering from
the model trained on the expanded standard corpus if that model is materially
different.

### M3. Critical constants remain untuned

`c_puct=1.5`, policy/value loss balance, optimizer schedule, and ensemble use
have not received clean equal-time strength ablations. The 32/64 simulation
choice was based on throughput and target diagnostics, not downstream Elo.
These are reasonable provisional values, not settled design choices. Tune a
small number in successive-halving experiments after the fixed corpus exists;
do not launch a combinatorial sweep.

### M4. Tree reuse was not assessed

Every move creates a fresh root (`src/breakthrough_zero/selfplay.py:176,286`).
The project carefully benchmarked state cloning versus replay versus
make/unmake, but did not benchmark retaining the selected search subtree. Tree
reuse may or may not be worthwhile once inherited-visit target semantics are
handled correctly. Under limited compute it deserves one bounded benchmark,
not an assumption.

### M5. Training reproducibility is incomplete

Seeds are set, but deterministic TensorFlow operations are not requested or
reported. Training records no end-to-end elapsed time, Slurm job ID/node,
output checkpoint hash, or accelerator model in `run.json`. Model snapshots and
optimizer-state policy are absent. Matched seeds therefore do not imply exact
reproducibility.

### M6. Local green tests omit the neural boundary

The README installs only the base package (`README.md:154`), while the three
network tests skip without TensorFlow (`tests/test_network.py:22`). There is no
GitHub CI. A student can run the advertised command, see `OK`, and miss every
Keras test.

Provide separate fast-core and full-TensorFlow commands, make skip counts
prominent, and add CPU TensorFlow CI plus the existing real-GPU HPC gate.

### M7. Documentation is not a reliable state dashboard

The README ends before the completed 256-game neural replay and generation-1
fine-tunes, does not link the replay release, and simultaneously documents the
bad padded mini boundary and the desired native boundary. Results, plans, and
current blockers are spread across phase files. The research-conclusion
evidence standard asks for links to raw data, logs, and checkpoints, but those
artifacts are often absent or HPC-only.

Add a short status table with `implemented`, `validated`, `invalidated`,
`running`, and `next gated action`; add a central experiment ledger with commit,
job, configuration, status, data/model hashes, and links. Phase reviews remain
detailed supporting documents, not the project-control surface.

### M8. The literature review is not complete enough to close design choices

The teacher later clarified that the remembered low-resource source was
OLIVAW, not a paper titled "AlphaZero on a Shoestring." The literature survey
now reviews OLIVAW directly. It also records modern implementations and the
practitioner failure report as a diagnostic lead; this finding is closed.

## Authorization gates

### Gate A: engineering boundary

- Native 5x5/75 and unchanged 8x8/192 models pass fail-closed CPU and GPU tests.
- Rules-derived maximum game length replaces 128 as the hard safety limit.
- Fixed-dataset compatibility and numerical-statistic audits are executable.
- Working tree is clean, CI is green, and the exact job script is committed.

### Gate B: reproducible research artifacts

- Existing raw data, all comparison checkpoints, run manifests, and logs have
  checksummed GitHub release entries and a README artifact index.
- Training and arena artifacts contain output/model hashes, full configs,
  actual timing, hardware, code state, and opening-suite hash.
- A canonical standard and mini opening suite is frozen before longitudinal
  ratings resume.

### Gate C: corrected fixed-data experiments

- Native mini uses the old 64 games only for a labelled smoke test, followed by
  a staged learning curve on a larger mini archive.
- Expanded standard data is generated in audited stages and analyzed through
  nested learning curves.
- Architecture comparisons have equal-time as their primary budget; all-four
  symmetry coverage and validation are measured rather than assumed.
- Best/last snapshots are preserved under a predeclared selection rule.

### Gate D: before generation 2

- The pending pretrained-versus-generation-1 arena is rerun under corrected
  identity, opening, timing, and interval rules.
- Replay mixture, capacity, age, and reuse are implemented and recorded.
- Raw-versus-ensemble, `c_puct`, and--only if warranted--root-noise experiments
  resolve the current provisional choices.
- A regression triggers diagnosis and rollback, never automatic extra training.

## Boss's decision

Authorize code fixes, tests, artifact publication, the native-mini smoke test,
and small audited data-generation stages. Do **not** authorize a long neural
self-play loop, generation 2, a promoted native-mini model trained only on 64
games, or definitive Elo/research claims from the current confidence intervals.

The project is recoverable without a rewrite. Its clean absolute-value core,
raw-data boundary, and saved negative results are worth preserving. The next
milestone is not "more hours on the GPU." It is making the written experimental
discipline true in executable code.
