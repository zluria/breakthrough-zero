# Mini Breakthrough tournament

Smoke ratings are anchored at random = 1000. Do not treat a small run
as a stable strength claim. Confidence intervals are head-to-head, not
global-rating intervals.

Termination counts: `{'terminal': 896}`. 
Abnormal games: **0**.

| Agent | Fitted Elo |
| --- | ---: |
| alpha-beta | 1736.6 |
| puct-tactical | 1719.1 |
| base-soft-z | 1569.6 |
| base-outcome | 1513.3 |
| tiny-soft-z | 1512.7 |
| puct-rollout | 1499.5 |
| tiny-outcome | 1449.5 |
| random | 1000.0 |

| Match (first-agent view) | W-L-D | Pair sweeps | Color splits | Elo | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| random vs alpha-beta | 0-32-0 | 0-16 | 0 | -607.4 | [-1004.1, -210.7] |
| random vs puct-rollout | 0-32-0 | 0-16 | 0 | -607.4 | [-1004.1, -210.7] |
| random vs puct-tactical | 1-31-0 | 0-15 | 1 | -481.6 | [-790.4, -172.9] |
| random vs tiny-outcome | 0-32-0 | 0-16 | 0 | -607.4 | [-1004.1, -210.7] |
| random vs tiny-soft-z | 0-32-0 | 0-16 | 0 | -607.4 | [-1004.1, -210.7] |
| random vs base-outcome | 1-31-0 | 0-15 | 1 | -481.6 | [-790.4, -172.9] |
| random vs base-soft-z | 0-32-0 | 0-16 | 0 | -607.4 | [-1004.1, -210.7] |
| alpha-beta vs puct-rollout | 28-4-0 | 12-0 | 4 | +305.4 | [+86.9, +523.8] |
| alpha-beta vs puct-tactical | 13-19-0 | 2-5 | 9 | -62.0 | [-223.8, +99.9] |
| alpha-beta vs tiny-outcome | 28-4-0 | 12-0 | 4 | +305.4 | [+86.9, +523.8] |
| alpha-beta vs tiny-soft-z | 27-5-0 | 11-0 | 5 | +267.6 | [+63.0, +472.2] |
| alpha-beta vs base-outcome | 26-6-0 | 10-0 | 6 | +234.5 | [+40.5, +428.5] |
| alpha-beta vs base-soft-z | 26-6-0 | 10-0 | 6 | +234.5 | [+40.5, +428.5] |
| puct-rollout vs puct-tactical | 5-27-0 | 0-11 | 5 | -267.6 | [-472.2, -63.0] |
| puct-rollout vs tiny-outcome | 22-10-0 | 8-2 | 6 | +128.1 | [-41.6, +297.8] |
| puct-rollout vs tiny-soft-z | 15-17-0 | 4-5 | 7 | -20.5 | [-180.2, +139.3] |
| puct-rollout vs base-outcome | 11-21-0 | 2-7 | 7 | -105.3 | [-271.7, +61.1] |
| puct-rollout vs base-soft-z | 14-18-0 | 2-4 | 10 | -41.1 | [-201.6, +119.5] |
| puct-tactical vs tiny-outcome | 29-3-0 | 13-0 | 3 | +350.0 | [+112.6, +587.4] |
| puct-tactical vs tiny-soft-z | 26-6-0 | 10-0 | 6 | +234.5 | [+40.5, +428.5] |
| puct-tactical vs base-outcome | 25-7-0 | 9-0 | 7 | +204.8 | [+19.0, +390.5] |
| puct-tactical vs base-soft-z | 23-9-0 | 7-0 | 9 | +152.1 | [-21.8, +326.0] |
| tiny-outcome vs tiny-soft-z | 12-20-0 | 2-6 | 8 | -83.3 | [-247.1, +80.5] |
| tiny-outcome vs base-outcome | 13-19-0 | 3-6 | 7 | -62.0 | [-223.8, +99.9] |
| tiny-outcome vs base-soft-z | 12-20-0 | 1-5 | 10 | -83.3 | [-247.1, +80.5] |
| tiny-soft-z vs base-outcome | 17-15-0 | 3-2 | 11 | +20.5 | [-139.3, +180.2] |
| tiny-soft-z vs base-soft-z | 10-22-0 | 1-7 | 8 | -128.1 | [-297.8, +41.6] |
| base-outcome vs base-soft-z | 12-20-0 | 1-5 | 10 | -83.3 | [-247.1, +80.5] |
