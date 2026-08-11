# Phase 30 review: first native-mini neural arena

## Purpose

This is a failure screen for six pretrained native 5x5 models, not a final
tournament. Each training run contributes one periodic checkpoint selected by
minimum validation objective within that run. The selector verifies checkpoint
hashes and the shared data/split/time contract. It does not compare raw losses
between different value targets.

Checkpoint selection is a separate one-CPU job with an `afterok` dependency
on all six learners. The GPU arena array is in turn dependent on selection, so
incomplete runs or hash failures allocate no evaluation accelerators.

Each model separately joins the same four fixed anchors: random, alpha-beta,
plain-rollout PUCT, and tactical-rollout PUCT. Every matchup uses 24 fresh
four-ply opening pairs with colors reversed, 50 ms of internal search time per
move, 50 ms scheduling grace, `c_puct=1.5`, and no search noise. Model inference
uses one orientation; four-way symmetry averaging remains a later equal-time
ablation rather than a free advantage.

## What 24 pairs can decide

The screen can reject grossly broken agents, time/replay failures, severe
color dependence, and models that are plainly weaker than all meaningful
anchors. It is specifically designed to expose the warned failure pattern in
which rollout-MCTS remains strongest and learning makes the search agent
worse.

The screen cannot establish a fine Elo ordering among six models. Compare
their direct records against the same tactical teacher and alpha-beta anchor,
including paired confidence intervals. Advance at most two coherent models to
a larger fresh-opening confirmation. Do not choose a winner from fitted Elo
point estimates across separate tasks.

## Stop conditions

- Any forfeit, agent exception, illegal move, or nonterminal ply-limit result.
- Missing/mismatched model or data hash, dirty worktree, or ruleset mismatch.
- A value head whose calibration or color behavior contradicts its training
  diagnostics.
- Both architectures/most targets clearly regress against the teacher: stop
  and diagnose data, loss balance, and optimization before neural self-play.

The raw 512-game corpus remains immutable regardless of the result, so a bad
screen costs no new self-play and can be retrained with a corrected learner.
