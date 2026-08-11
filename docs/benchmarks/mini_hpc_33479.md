# Duplicate-opening mini tournament: HPC job 33479

This diagnostic tournament replaced PUCT/Dirichlet opening generation with
four uniform-random plies: exactly two moves per side. Every matchup used the
same 32 saved starts and played each start with colors reversed. Raw evidence
is in [`../../results/mini-hpc-33479/`](../../results/mini-hpc-33479/).

## Reproducibility and resources

- Commit: `2b2b07da2b9fc9cf441666b1370725aee8d08e08`
- Slurm job: `33479`, exit `0:0`, elapsed 98 seconds
- Host: `HPC-RTX3070-08`
- Allocation: CPU-only, 2 GB requested, 30 MB peak RSS, no GPU TRES
- Match: 32 opening pairs per matchup, 50 ms per rated move
- Scope: one opening seed and one tournament run

The job passed all 67 tests before playing. All 384 games replay to their exact
terminal winner. There were no draws, forfeits, illegal moves, agent errors, or
ply-limit results. All openings are distinct, nonterminal, and have no
immediate winning move. The slowest move took 63.95 ms, below the predeclared
70 ms forfeit threshold.

## Results

Random is fixed at 1000 in the fitted pool:

| Agent | Fitted Elo |
| --- | ---: |
| Tactical-rollout PUCT | 1754.4 |
| Alpha-beta | 1677.9 |
| Plain-rollout PUCT | 1497.0 |
| Random | 1000.0 |

Pair sweeps count starts won by one agent with both colors. A color split means
the same absolute color won both games, so each agent scored one point.

| First-agent view | W-L-D | Pair sweeps | Color splits | Elo | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| random vs alpha-beta | 1-63-0 | 0-31 | 1 | -602.1 | [-907.1, -297.0] |
| random vs plain PUCT | 1-63-0 | 0-31 | 1 | -602.1 | [-907.1, -297.0] |
| random vs tactical PUCT | 0-64-0 | 0-32 | 0 | -725.2 | [-1119.8, -330.5] |
| alpha-beta vs plain PUCT | 52-12-0 | 20-0 | 12 | +244.1 | [+99.3, +388.9] |
| alpha-beta vs tactical PUCT | 26-38-0 | 3-9 | 20 | -63.9 | [-182.1, +54.4] |
| plain vs tactical PUCT | 9-55-0 | 0-23 | 9 | -299.3 | [-458.9, -139.6] |

The interval treats one opening pair as the independent observation and adds
one explicit virtual drawn pair for finite early Elo.

## What the two random wins mean

Random's wins were on openings 27 against alpha-beta and 6 against plain PUCT.
Both were Player 1 wins, and Player 1 also won the color-reversed paired game.
A deeper alpha-beta check proved Player 1 wins: opening 27 at depth 3 (239
nodes) and opening 6 at depth 5 (4,156 nodes). Random happened to follow the
winning line; it swept no pair.

This is not a reason to filter every engine-proven biased opening. That would
make the opening generator depend on an agent and could bias evaluation.
Duplicate color reversal already cancels the seat advantage and exposes it in
the cross-table. We reject only terminal and immediate-win starts.

## Interpretation

Tactical rollouts beat plain rollouts at equal time by about 299 Elo in this
run, contrary to the pre-registered prediction that their throughput cost
would erase the benefit. Alpha-beta also beat plain PUCT. Tactical PUCT led
alpha-beta by about 64 Elo, but its interval crosses zero.

These are preliminary mini-board results, not claims about standard 8x8 play or
trained neural agents. Because opening depth and generator changed together
from job 33478, the rating change between jobs does not isolate a causal noise
effect.
