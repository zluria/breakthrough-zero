# Breakthrough tournament

Smoke ratings are anchored at random = 1000. Do not treat a small run
as a stable strength claim. Confidence intervals are head-to-head, not
global-rating intervals.

Termination counts: `{'terminal': 448}`. 
Abnormal games: **0**.

| Agent | Fitted Elo |
| --- | ---: |
| alpha-beta | 1791.8 |
| puct-tactical | 1492.5 |
| s64-soft-z | 1428.8 |
| s64-outcome | 1406.5 |
| puct-rollout | 1383.6 |
| s32-soft-z | 1291.4 |
| s32-outcome | 1225.6 |
| random | 1000.0 |

| Match (first-agent view) | W-L-D | Pair sweeps | Color splits | Elo | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| random vs alpha-beta | 0-16-0 | 0-8 | 0 | -492.2 | [-892.8, -91.6] |
| random vs puct-rollout | 0-16-0 | 0-8 | 0 | -492.2 | [-892.8, -91.6] |
| random vs puct-tactical | 0-16-0 | 0-8 | 0 | -492.2 | [-892.8, -91.6] |
| random vs s32-outcome | 2-14-0 | 0-6 | 2 | -279.6 | [-554.5, -4.7] |
| random vs s32-soft-z | 0-16-0 | 0-8 | 0 | -492.2 | [-892.8, -91.6] |
| random vs s64-outcome | 2-14-0 | 0-6 | 2 | -279.6 | [-554.5, -4.7] |
| random vs s64-soft-z | 0-16-0 | 0-8 | 0 | -492.2 | [-892.8, -91.6] |
| alpha-beta vs puct-rollout | 16-0-0 | 8-0 | 0 | +492.2 | [+91.6, +892.8] |
| alpha-beta vs puct-tactical | 16-0-0 | 8-0 | 0 | +492.2 | [+91.6, +892.8] |
| alpha-beta vs s32-outcome | 16-0-0 | 8-0 | 0 | +492.2 | [+91.6, +892.8] |
| alpha-beta vs s32-soft-z | 16-0-0 | 8-0 | 0 | +492.2 | [+91.6, +892.8] |
| alpha-beta vs s64-outcome | 15-1-0 | 7-0 | 1 | +361.2 | [+45.4, +677.0] |
| alpha-beta vs s64-soft-z | 16-0-0 | 8-0 | 0 | +492.2 | [+91.6, +892.8] |
| puct-rollout vs puct-tactical | 3-13-0 | 0-5 | 3 | -217.6 | [-468.3, +33.0] |
| puct-rollout vs s32-outcome | 13-3-0 | 5-0 | 3 | +217.6 | [-33.0, +468.3] |
| puct-rollout vs s32-soft-z | 12-4-0 | 4-0 | 4 | +166.0 | [-69.1, +401.0] |
| puct-rollout vs s64-outcome | 7-9-0 | 1-2 | 5 | -38.8 | [-253.3, +175.7] |
| puct-rollout vs s64-soft-z | 6-10-0 | 1-3 | 4 | -78.5 | [-296.7, +139.7] |
| puct-tactical vs s32-outcome | 12-4-0 | 4-0 | 4 | +166.0 | [-69.1, +401.0] |
| puct-tactical vs s32-soft-z | 14-2-0 | 6-0 | 2 | +279.6 | [+4.7, +554.5] |
| puct-tactical vs s64-outcome | 13-3-0 | 5-0 | 3 | +217.6 | [-33.0, +468.3] |
| puct-tactical vs s64-soft-z | 9-7-0 | 2-1 | 5 | +38.8 | [-175.7, +253.3] |
| s32-outcome vs s32-soft-z | 9-7-0 | 2-1 | 5 | +38.8 | [-175.7, +253.3] |
| s32-outcome vs s64-outcome | 0-16-0 | 0-8 | 0 | -492.2 | [-892.8, -91.6] |
| s32-outcome vs s64-soft-z | 4-12-0 | 0-4 | 4 | -166.0 | [-401.0, +69.1] |
| s32-soft-z vs s64-outcome | 4-12-0 | 0-4 | 4 | -166.0 | [-401.0, +69.1] |
| s32-soft-z vs s64-soft-z | 7-9-0 | 0-1 | 7 | -38.8 | [-253.3, +175.7] |
| s64-outcome vs s64-soft-z | 4-12-0 | 0-4 | 4 | -166.0 | [-401.0, +69.1] |
