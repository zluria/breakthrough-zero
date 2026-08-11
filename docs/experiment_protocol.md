# Experiment protocol

The aim is to measure which changes improve playing strength per unit of real
compute, not which configuration was allowed to run more iterations.

## Training comparisons

An ablation changes one named factor. Compared agents must use:

- The same hardware type and exclusive resource allocation.
- The same wall-clock training budget, including self-play and optimization.
- The same starting checkpoint and immutable starting data when applicable.
- Matched random seeds, data split, evaluation schedule, and opponent pool.
- The same rules, adjudication, and software version outside the named change.

We report wall time as the primary budget and also record positions, search
nodes, neural evaluations, optimizer steps, and energy or scheduler accounting
when available. Those secondary counts explain *why* a method is faster or
slower; they do not replace the wall-clock comparison.

If two methods require materially different one-time preprocessing, both
end-to-end time and steady-state time are reported. Failed and interrupted runs
remain in the experiment log.

## Playing-strength comparisons

An evaluation game gives both agents the same wall-clock allowance per move on
the same machine class. Search implementations receive a short untimed warm-up
before clocks begin so tracing, imports, and device initialization do not favor
the agent that moves first.

Games are paired: each sampled opening is played twice with colors reversed.
The opening set and pair order are generated before the match and saved. When
agents share stochastic components, independent recorded seeds are used.

The arena records move time, nodes, neural evaluations, result, color, opening,
agent/model hashes, hardware, and software revision. Any timeout policy is
fixed before the match.

## Elo and uncertainty

The primary head-to-head report contains wins, losses, draws, score, Elo
difference, and a 95% confidence interval. Breakthrough normally has no draws,
but the format permits them for timeouts or future rule variants. Color and
opening-pair effects are reported separately when large enough to diagnose.

A rating pool is anchored by immutable baselines (random, dummy MCTS,
alpha-beta, and named neural checkpoints). Pool ratings are useful summaries;
paired head-to-head results are the evidence for an ablation. We do not promote
a variant on a point estimate whose uncertainty overlaps a practically
irrelevant effect.

Sequential tests may stop a clearly decisive match early only when their
boundaries and minimum sample count were fixed in advance. Otherwise match
length is fixed before results are viewed.

## Repetition and claims

Pilot experiments may use one seed to reject obvious failures. A positive
research claim requires repeated training seeds when the cost is affordable,
or must be labeled preliminary. Every conclusion links to its configuration,
raw arena games, logs, and uncertainty calculation.
