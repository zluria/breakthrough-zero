# Dummy self-play pilot

Date: 2026-08-11. Environment: local Windows Codex Python runtime. Rules:
5x5 with one starting row. These are pipeline and routing checks, not
playing-strength evidence.

## End-to-end durability gate

Command shape:

`python scripts/generate_dummy_selfplay.py <output> --games 2 --chunk-games 1 --simulations 8 --rules mini --seed 20260811`

The first invocation wrote two games, 31 positions, two NPZ files, and two
checksummed manifests in 1.05 seconds. The second invocation loaded and
verified both chunks, including configuration, game-index ranges, and stored
seeds; it generated zero positions.

## Optional tactical rollout timing

Both runs used master seed 417, six mini games, 16 simulations per move, and no
root noise.

| Rollout selector | Games | Positions | Seconds | Positions/s |
| --- | ---: | ---: | ---: | ---: |
| Uniform bit sampler | 6 | 84 | 1.85 | 45.52 |
| Prefer win, then capture | 6 | 83 | 1.94 | 42.68 |

The tactical selector was about 6% slower per generated position here. Six
games are not enough to distinguish stable overhead from timing noise, and no
playing-strength match was run. Decision: retain it as an explicit ablation;
keep uniform rollouts as the dummy evaluator's default.
