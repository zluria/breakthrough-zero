# Mini Breakthrough tournament

Smoke ratings are anchored at random = 1000. Do not treat a small run
as a stable strength claim. Confidence intervals are head-to-head, not
global-rating intervals.

| Agent | Fitted Elo |
| --- | ---: |
| alpha-beta | 1673.2 |
| puct-tactical | 1659.2 |
| puct-rollout | 1528.1 |
| random | 1000.0 |

| Match (first-agent view) | W-L-D | Score | Elo | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| random vs alpha-beta | 1-63-0 | 0.016 | -650.7 | [-907.8, -393.5] |
| random vs puct-rollout | 2-62-0 | 0.031 | -559.2 | [-766.3, -352.1] |
| random vs puct-tactical | 1-63-0 | 0.016 | -650.7 | [-907.8, -393.5] |
| alpha-beta vs puct-rollout | 46-18-0 | 0.719 | +160.1 | [+67.6, +252.6] |
| alpha-beta vs puct-tactical | 34-30-0 | 0.531 | +21.4 | [-62.4, +105.2] |
| puct-rollout vs puct-tactical | 19-45-0 | 0.297 | -147.2 | [-238.3, -56.1] |
