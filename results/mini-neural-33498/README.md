# Mini Breakthrough tournament

> **Invalid as a strength result.** A post-run trajectory audit found 148
> `time_forfeit` games, mostly from rollout PUCT starting an indivisible
> simulation too close to its deadline.  The files are retained as debugging
> evidence.  Do not compare or publish the Elo values below.

Smoke ratings are anchored at random = 1000. Do not treat a small run
as a stable strength claim. Confidence intervals are head-to-head, not
global-rating intervals.

| Agent | Fitted Elo |
| --- | ---: |
| alpha-beta | 1622.2 |
| base-soft-z | 1446.3 |
| tiny-soft-z | 1413.0 |
| base-outcome | 1403.2 |
| puct-tactical | 1347.1 |
| tiny-outcome | 1344.0 |
| puct-rollout | 1288.1 |
| random | 1000.0 |

| Match (first-agent view) | W-L-D | Pair sweeps | Color splits | Elo | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| random vs alpha-beta | 0-32-0 | 0-16 | 0 | -607.4 | [-1004.1, -210.7] |
| random vs puct-rollout | 9-23-0 | 0-7 | 9 | -152.1 | [-326.0, +21.8] |
| random vs puct-tactical | 10-22-0 | 0-6 | 10 | -128.1 | [-297.8, +41.6] |
| random vs tiny-outcome | 2-30-0 | 0-14 | 2 | -405.7 | [-670.5, -140.9] |
| random vs tiny-soft-z | 1-31-0 | 0-15 | 1 | -481.6 | [-790.4, -172.9] |
| random vs base-outcome | 1-31-0 | 0-15 | 1 | -481.6 | [-790.4, -172.9] |
| random vs base-soft-z | 0-32-0 | 0-16 | 0 | -607.4 | [-1004.1, -210.7] |
| alpha-beta vs puct-rollout | 29-3-0 | 14-1 | 1 | +350.0 | [+112.6, +587.4] |
| alpha-beta vs puct-tactical | 23-9-0 | 7-0 | 9 | +152.1 | [-21.8, +326.0] |
| alpha-beta vs tiny-outcome | 27-5-0 | 12-1 | 3 | +267.6 | [+63.0, +472.2] |
| alpha-beta vs tiny-soft-z | 26-6-0 | 10-0 | 6 | +234.5 | [+40.5, +428.5] |
| alpha-beta vs base-outcome | 27-5-0 | 11-0 | 5 | +267.6 | [+63.0, +472.2] |
| alpha-beta vs base-soft-z | 26-6-0 | 10-0 | 6 | +234.5 | [+40.5, +428.5] |
| puct-rollout vs puct-tactical | 14-18-0 | 4-6 | 6 | -41.1 | [-201.6, +119.5] |
| puct-rollout vs tiny-outcome | 17-15-0 | 3-2 | 11 | +20.5 | [-139.3, +180.2] |
| puct-rollout vs tiny-soft-z | 9-23-0 | 1-8 | 7 | -152.1 | [-326.0, +21.8] |
| puct-rollout vs base-outcome | 11-21-0 | 1-6 | 9 | -105.3 | [-271.7, +61.1] |
| puct-rollout vs base-soft-z | 12-20-0 | 2-6 | 8 | -83.3 | [-247.1, +80.5] |
| puct-tactical vs tiny-outcome | 17-15-0 | 2-1 | 13 | +20.5 | [-139.3, +180.2] |
| puct-tactical vs tiny-soft-z | 17-15-0 | 3-2 | 11 | +20.5 | [-139.3, +180.2] |
| puct-tactical vs base-outcome | 13-19-0 | 1-4 | 11 | -62.0 | [-223.8, +99.9] |
| puct-tactical vs base-soft-z | 12-20-0 | 2-6 | 8 | -83.3 | [-247.1, +80.5] |
| tiny-outcome vs tiny-soft-z | 15-17-0 | 2-3 | 11 | -20.5 | [-180.2, +139.3] |
| tiny-outcome vs base-outcome | 12-20-0 | 2-6 | 8 | -83.3 | [-247.1, +80.5] |
| tiny-outcome vs base-soft-z | 11-21-0 | 1-6 | 9 | -105.3 | [-271.7, +61.1] |
| tiny-soft-z vs base-outcome | 18-14-0 | 4-2 | 10 | +41.1 | [-119.5, +201.6] |
| tiny-soft-z vs base-soft-z | 16-16-0 | 2-2 | 12 | +0.0 | [-159.5, +159.5] |
| base-outcome vs base-soft-z | 13-19-0 | 1-4 | 11 | -62.0 | [-223.8, +99.9] |
