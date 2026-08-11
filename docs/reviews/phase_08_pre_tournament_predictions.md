# Phase 8 review: pre-tournament predictions

Recorded before any Elo tournament result was observed on 2026-08-11.

## Mini-board prediction

At an adequate and equal wall-clock allowance per move, the expected order is:

1. alpha-beta;
2. uniform-policy PUCT with random-rollout value;
3. uniform random.

Alpha-beta is predicted to lead dummy PUCT by roughly 150--350 Elo. Dummy PUCT
is predicted to lead random by roughly 100--250 Elo. Random is predicted to
lose every game against alpha-beta in the first 5x5 match.

The tactical rollout variant is predicted to tie or slightly underperform the
plain rollout at fixed wall time: its choice quality may improve, but the pilot
measured lower throughput. Alpha-beta versus PUCT is the least certain ranking.

## What would change our minds

A random win is replayed before interpretation. We first inspect timeout or
illegal-move forfeits, opening/color imbalance, simulations completed, and the
terminal last-mover invariant. A genuine legal win then counts normally.

The first small run is a smoke estimate, not a stable rating. Promotion to a
research claim requires a predeclared larger match, uncertainty intervals, and
the same saved openings played with colors reversed.

## Gate and resource review

The arena passed 64 tests before the local smoke run. Tests cover exact opening
replay, nonterminal prefixes, color reversal, equal supplied budgets, matched
agent seeds, terminal replay, absolute winners, and explicit timeout and
illegal-move forfeits. The smoke run then completed 24/24 games by the rules;
details are in [`../benchmarks/mini_arena_smoke_20260811.md`](../benchmarks/mini_arena_smoke_20260811.md).

The main runtime bottleneck is rated search, not opening generation or rating
arithmetic. The first HPC tournament therefore uses one CPU, 2 GB of memory,
and no GPU allocation. It uses 32 saved opening pairs per matchup at 50 ms per
move. Stop and diagnose rather than quote ratings if tests fail, any forfeit
occurs, pair balance fails, or recorded times show systematic budget overruns.

Job `33477` stopped at the test gate in zero scheduler seconds because shell
activation selected the cluster's system Python 3.9, which lacked NumPy. It
played no games. The retry uses the absolute Python 3.11 executable already
proven by job `33476` and explicitly excludes nodes RTX3070-06 and -07.

Job `33478` completed, but its audit found that a six-ply prefix is too deep
for this mini game: one opening gave the mover an immediate win. Four random
wins all followed the absolute color that also won the paired game. The
preserved result and diagnosis are in
[`../benchmarks/mini_hpc_33478.md`](../benchmarks/mini_hpc_33478.md).

The separately labeled diagnostic rerun changes opening generation to the
simpler proposal motivated by duplicate chess and bridge events: two uniform-
random moves per side, shared by every matchup, with no Dirichlet search noise.
Its ratings also use opening pairs, rather than their two correlated games, as
independent observations for uncertainty.

Job `33479` completed 384 games with no forfeits or replay failures. Random
swept no pair. Tactical PUCT beat plain PUCT decisively; alpha-beta beat plain
PUCT; tactical PUCT versus alpha-beta remained uncertain. The full audit is in
[`../benchmarks/mini_hpc_33479.md`](../benchmarks/mini_hpc_33479.md). The arena
gate is complete; fixed-data neural experiments are next.
