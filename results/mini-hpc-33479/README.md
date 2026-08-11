# Mini Breakthrough tournament

Smoke ratings are anchored at random = 1000. Do not treat a small run
as a stable strength claim. Confidence intervals are head-to-head, not
global-rating intervals.

| Agent | Fitted Elo |
| --- | ---: |
| puct-tactical | 1754.4 |
| alpha-beta | 1677.9 |
| puct-rollout | 1497.0 |
| random | 1000.0 |

| Match (first-agent view) | W-L-D | Pair sweeps | Color splits | Elo | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| random vs alpha-beta | 1-63-0 | 0-31 | 1 | -602.1 | [-907.1, -297.0] |
| random vs puct-rollout | 1-63-0 | 0-31 | 1 | -602.1 | [-907.1, -297.0] |
| random vs puct-tactical | 0-64-0 | 0-32 | 0 | -725.2 | [-1119.8, -330.5] |
| alpha-beta vs puct-rollout | 52-12-0 | 20-0 | 12 | +244.1 | [+99.3, +388.9] |
| alpha-beta vs puct-tactical | 26-38-0 | 3-9 | 20 | -63.9 | [-182.1, +54.4] |
| puct-rollout vs puct-tactical | 9-55-0 | 0-23 | 9 | -299.3 | [-458.9, -139.6] |
