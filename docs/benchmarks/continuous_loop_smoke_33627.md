# Continuous-loop boundary: jobs 33625--33627

## Purpose

Exercise one complete noise-off transition from a saved actor through neural
self-play, corpus audit, bounded replay training, optimizer checkpointing, and
publication of the next actor before allocating an hour.

## Fail-closed findings

- Job 33625 stopped in tests because the isolated review directory mixed an
  older tournament test with an older repository script. It generated no data.
- Job 33626 exposed an exact-duration edge: milliseconds of preflight made a
  90-second smoke with a 90-second first-cycle reserve start zero cycles. It
  generated no data. The helper now allows only a one-second startup tolerance
  at that boundary while retaining the full reserve in real runs.

These were harness failures, not evidence about learning strength.

## Passing result

Job 33627 completed on an RTX 3070 in 30 seconds:

- targeted Python and real TensorFlow boundary tests passed;
- 64 noise-off games were generated from generation 2 and audited;
- replay training used one fresh archive and the fixed historical source;
- replay consumption was 3.721, below the hard limit of 4;
- the actor changed from
  `fcab56f571c1392443871d6b012c2307ed42b493ed88c2fe3d7a4e35b1501d4d`
  to `032ad6199234d786ecee32ad12af02268364fd141178cc5bfe0cfcd42932501d`;
- `progress.json` reached `complete`, with noise fraction exactly zero.

This authorizes the one-hour operational validation in phase 37. It is not an
Elo result and does not establish the replay ratio, target, or architecture as
optimal.
