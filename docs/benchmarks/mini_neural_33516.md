# Valid mini-board neural screening: job 33516

This is the first valid tournament containing the pretrained Keras agents.
It is a screening experiment on the 5x5 debug game, not a final strength
claim.  Every matchup used the same 16 saved four-ply openings, once with each
color, for 32 games.  Rated search had no noise and each non-random agent was
given a nominal 50 ms per move.  The arena allowed 100 ms of scheduler grace
before declaring a forfeit; the grace was not passed to the agents.

All **896 games ended at a rules terminal**.  There were no time forfeits,
illegal moves, agent exceptions, ply-limit draws, malformed pairs, or failed
color reversals.  The raw games and generated summary are in
[`results/mini-neural-33516`](../../results/mini-neural-33516/README.md).

## Global screening ratings

These fitted ratings are anchored at random = 1000.  Head-to-head intervals,
not the fitted global order, determine whether a comparison is convincing.

| Agent | Fitted Elo |
| --- | ---: |
| Alpha-beta | 1736.6 |
| Tactical-rollout PUCT | 1719.1 |
| 64x4 soft-Z network | 1569.6 |
| 64x4 outcome network | 1513.3 |
| 32x3 soft-Z network | 1512.7 |
| Plain-rollout PUCT | 1499.5 |
| 32x3 outcome network | 1449.5 |
| Random | 1000.0 |

Random lost 222 of its 224 games.  This is a useful end-to-end sanity check,
but rating differences against the saturated random anchor are not precise.

## Comparisons that answer design questions

| First agent vs second | W-L-D | Elo difference | 95% CI |
| --- | ---: | ---: | ---: |
| Tactical PUCT vs plain PUCT | 27-5-0 | +267.6 | [+63.0, +472.2] |
| Tactical PUCT vs alpha-beta | 19-13-0 | +62.0 | [-99.9, +223.8] |
| 64x4 soft-Z vs 64x4 outcome | 20-12-0 | +83.3 | [-80.5, +247.1] |
| 32x3 soft-Z vs 32x3 outcome | 20-12-0 | +83.3 | [-80.5, +247.1] |
| 64x4 soft-Z vs 32x3 soft-Z | 22-10-0 | +128.1 | [-41.6, +297.8] |
| 64x4 outcome vs 32x3 outcome | 19-13-0 | +62.0 | [-99.9, +223.8] |

The soft-Z target won 20-12 in both controlled architecture comparisons.
That repeated direction agrees with its much lower validation error, but each
playing-strength interval still includes zero.  The larger network also led in
both target-controlled comparisons without clearing the screening interval.
Neither choice is therefore called confirmed yet.

Pretraining has not made the neural agents stronger than the best classical
searches.  The 64x4 soft-Z agent lost 9-23 to tactical PUCT (-152 Elo, 95% CI
[-326, +22]) and 6-26 to alpha-beta (-235 Elo, 95% CI [-429, -41]).  This is
the expected diagnostic checkpoint before self-play, not a reason to hide or
discard the result.

## Timing and work audit

| Agent | Moves | Mean elapsed (s) | 95th percentile (s) | Mean work units |
| --- | ---: | ---: | ---: | ---: |
| Alpha-beta | 954 | 0.02577 | 0.05005 | 2396.1 nodes |
| Tactical-rollout PUCT | 1035 | 0.05570 | 0.10675 | 2241.7 simulations |
| Plain-rollout PUCT | 929 | 0.05472 | 0.10627 | 2380.0 simulations |
| 64x4 outcome | 941 | 0.04584 | 0.04675 | 5.7 simulations |
| 64x4 soft-Z | 975 | 0.04596 | 0.04671 | 5.4 simulations |
| 32x3 outcome | 967 | 0.03913 | 0.03958 | 5.2 simulations |
| 32x3 soft-Z | 941 | 0.03905 | 0.03953 | 16.9 simulations |

PUCT never receives the grace as search time.  A few CPU rollout calls were
descheduled and returned late; those records had unusually *low* work counts,
so the longer measured time did not buy more search.  Alpha-beta sometimes
returned early after proving a terminal value.  Neural PUCT stops before
starting an inference that its recent simulation timing predicts will cross
the deadline, so it deliberately leaves a few milliseconds unused.

## Decision

Promote the 64x4 soft-Z checkpoint as the provisional neural candidate.  Keep
the other three checkpoints and the immutable pretraining games.  First test
the same four networks on standard 8x8 under the identical paired protocol;
then tune search constants around the strongest standard checkpoint.  Do not
start expensive neural self-play until those fixed-data comparisons and the
search audit are complete.
