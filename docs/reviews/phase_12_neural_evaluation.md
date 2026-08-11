# Phase 12 review: first neural playing-strength evaluation

## Purpose

Training curves cannot establish playing strength.  The first neural models
will therefore enter the existing duplicate-opening arena without changing its
rules: four random opening plies, every position once with each color, equal
wall-clock time per move, no rated-search noise, saved trajectories, and
pair-level intervals.

The tournament includes random, alpha-beta, plain and tactical rollout PUCT,
and all four fixed-data neural PUCT agents.  This produces direct neural-versus-
baseline and neural-versus-neural comparisons from one fixed opening suite.

## Neural search boundary

The Keras evaluator returns an absolute Player-1 value and legal policy
probabilities.  The generic timed PUCT adapter uses the existing absolute-Q
search unchanged.  Model instances are loaded once and shared by fresh
per-game agent wrappers; loading and warm-up happen outside rated move clocks.

Inference is deliberately single-position in this first evaluation.  That is
not the final self-play architecture, but equal wall-clock games expose its
real cost.  We separately record simulations per move so that a weak result
can be separated into network quality and inference-throughput effects.

## Predictions and interpretation

1. Every neural agent should comfortably beat random.  Failure is a correctness
   alarm.
2. A neural agent may be weaker than tactical rollout PUCT at equal wall-clock
   because a batch-one Keras call permits fewer simulations.  Matched-visit
   diagnostics are required before calling that a learning failure.
3. Soft-Z should help value-guided search sooner, but it may also inherit search
   bias.  The result is genuinely uncertain.
4. The larger model may fit better yet play worse per second.  Elo per unit
   time, not parameter count or loss, decides promotion.

Sixteen opening pairs per matchup are a screening tournament.  Confidence
intervals, color splits, forfeits, and work counts will be reported.  Close
results must be repeated with more pairs before entering the conclusions file.

### Timing audit amendment

The first two attempts exposed cluster scheduling pauses.  With the improved
search timer, ordinary neural moves averaged 39--45 ms and rollout PUCT moves
50 ms, but a small repeatable subset was descheduled and returned in
100--130 ms with *less* search work.  Treating those pauses as game losses
confounded strength with operating-system scheduling.

The internal budget remains 50 ms.  The arena now uses 100 ms of forfeit grace,
so a move above 150 ms still fails the game and therefore the fail-closed job.
Actual elapsed time and simulations remain in every move record.  The grace is
not passed to the agent and cannot intentionally buy more search.
