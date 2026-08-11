# Phase 20 review: shorten opening visit sampling

## Why this is the next one-factor test

The symmetry-averaged 32-simulation pilot is the selected production search
budget: it generated 33.8 positions/second, had 42.2% Player-1 wins, and kept
all 64 trajectories and 4/8/12-ply prefixes unique. Doubling to 64 simulations
halved throughput for only small target changes.

Root noise remains off. The remaining exploration mechanism samples moves from
root visits for the first 12 plies (six moves per side). The teacher suggested
that two stochastic moves per side may be enough. The saved 32-simulation data
supports testing this directly: its 64 four-ply prefixes are already all
different.

## Comparison

Repeat the same 64 seeds, model, exact symmetry ensemble, 32 simulations,
`c_puct=1.5`, temperature 1, and zero root noise. Change only
`sample_until_ply` from 12 to 4. Since the first four sampled choices and their
RNG streams are identical, divergence after ply four is attributable to the
shorter schedule.

Choose four plies if trajectory/prefix diversity remains high and no target or
seat diagnostic materially worsens. This reduces gratuitous late opening
randomness and aligns self-play exploration with the duplicate-evaluation
opening design. Do not add Dirichlet noise while ordinary visit sampling is
already producing diverse games.
