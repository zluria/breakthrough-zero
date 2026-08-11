# Research conclusions

This file holds general, high-impact findings discovered by the project. The
README explains how the project works; this file explains what experiments
taught us beyond the assignment.

## Evidence standard

Each claimed finding must include:

- The question and one-factor comparison.
- Equal wall-clock budgets and hardware used.
- Training seeds and saved configuration identifiers.
- Paired, color-reversed arena games under equal wall-clock move limits.
- Wins/losses/draws, Elo difference, and a 95% confidence interval.
- Links to raw data, logs, and model checkpoints.
- Scope and important limitations.

Labels used below are **confirmed**, **preliminary**, **negative result**, and
**engineering observation**. Hypotheses belong in the experiment plan, not in
the conclusions.

## Findings

No playing-strength conclusions have been measured yet. Two local engineering
observations are established and must be reprofiled on the HPC:

- **Engineering observation:** bit-sampling one rollout move avoided building
  the full legal list and was about 6.6 times faster in the selector
  microbenchmark. Full uniform rollouts were about 3.5 times faster.
- **Engineering observation:** state strategy depends on search budget. Replay
  was about 10% faster at 32 simulations; lazy visited-node state caching was
  7--8% faster at 100 and 400. The full-search implementation uses lazy cache.

Raw commands and results are in
[`docs/benchmarks/foundation_hot_paths.md`](docs/benchmarks/foundation_hot_paths.md).

## Experiment register

| ID | Question | Status | Wall-clock budget | Result |
| --- | --- | --- | --- | --- |
| V001 | Does soft-Z beat final-result value training? | Planned | Set after HPC pilot | Pending |
| S001 | Does playout-cap randomization improve Elo per hour? | Planned | Set after HPC pilot | Pending |
| N001 | Does global pooling improve Elo per hour? | Deferred | Not set | Pending |
| N002 | Does an opponent-next-policy auxiliary head improve Elo per hour? | Deferred | Not set | Pending |
| T001 | Which compact CNN is strongest after equal-time training on fixed pretraining data? | Planned | Set after HPC pilot | Pending |
| T002 | Which optimizer and schedule are strongest after equal-time training on fixed data? | Planned | Set after HPC pilot | Pending |
| R001 | Do win/capture-preferred rollouts improve Elo per hour over uniform rollouts? | Planned | Set after HPC pilot | Pending |
| E001 | Which root-noise fraction and concentration improve Elo per hour? | Planned | Successive halving after diversity pilot | Pending |
| B001 | Which replay-window age best balances forgetting and staleness? | Planned | Equal-time short/medium/long windows | Pending |

The full fairness rules are in
[`docs/experiment_protocol.md`](docs/experiment_protocol.md).
