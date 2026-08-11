# Phase 16 review: first standard-board neural arena

## Gate and scope

This screening tournament is submitted with an `afterok` dependency on the
corrected mini neural arena.  If the mini run records even one abnormal game,
the standard evaluation will not start.

The standard tournament uses eight duplicate opening pairs, four random plies,
color reversal, a 50 ms internal move budget, 100 ms scheduler grace, and zero
search noise.  All trajectories, elapsed times, simulations, termination
reasons, and paired intervals are saved.  Any move beyond the grace or any
nonterminal completion fails the job after preserving its evidence.

## Agents

- Random, as the 1000-Elo anchor.
- Iterative-deepening alpha-beta.
- Plain and tactical rollout PUCT.
- Four 64-channel, four-block neural PUCT agents trained on equal data:
  32/64 simulations crossed with outcome/soft-Z value targets.

Models are loaded and warmed outside rated clocks.  Search remains batch-one,
so simulations per move are an important part of this first result.

## Questions and predictions

1. Do the standard neural agents beat random and at least approach rollout
   PUCT?  Failure against random triggers the regression ladder.
2. Does soft-Z's much lower held-out value error translate into Elo?
3. Does the more expensive 64-simulation dataset improve Elo enough to offset
   generation throughput dropping from about 30 to 18 positions per second?
4. Are differences actually policy quality, or merely batch-one inference
   throughput?  Actual simulations per move will be reported.

Eight pairs give only a screen.  The best candidates receive a larger direct
paired match; the full eight-agent table is not treated as a final ranking.
