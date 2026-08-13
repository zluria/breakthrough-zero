# Phase 37: one-hour continuous-loop validation

## Question

Does the repaired learner remain operational and avoid an obvious playing-
strength regression when self-play and training are connected continuously for
one hour?

This is a baseline health check, not an exploration experiment. Root
Dirichlet noise remains off. The proposed `0` versus `0.10/10` comparison waits
until this loop is demonstrably healthy.

## Migration boundary

The starting actor is generation 2, SHA-256
`fcab56f571c1392443871d6b012c2307ed42b493ed88c2fe3d7a4e35b1501d4d`.
It is the last coherent self-play-trained actor before the known-broken
large-data learner. The discarded models from that 71x weighting experiment
are not resurrected. Because the old checkpoint contains weights but no Adam
state, the first repaired update initializes Adam once; every later cycle
strictly restores its moments and global optimizer step.

## Frozen loop

- Native 5x5 32x3 CNN and absolute Player-1 values.
- 1,024 new games per cycle, 32 PUCT simulations, `c_puct=1.5`.
- Visit sampling through ply 4; deterministic thereafter.
- Root noise fraction 0 and recorded total concentration 10.
- The preserved 12,288-game generation-1 neural archive supplies exactly 25%
  of every replay batch. A sliding window of the newest four complete cycle
  archives supplies 75%.
- Batch 256, Adam `3e-4`, mixed-Z/Q target, full-action policy loss.
- At most 120 learner seconds and `R=4` optimizer examples per newly generated
  training position per cycle.
- A stable hash of each immutable game seed determines train versus validation;
  adding or evicting replay archives cannot move a game across that boundary.
- The newest completed checkpoint always becomes the next actor. Validation
  best is diagnostic only. No acceptance, rejection, promotion, or rollback.

The loop reserves enough measured time for a complete next cycle rather than
starting work that the one-hour budget cannot finish.

## Live diagnostics

`progress.json` is atomically replaced before and after every phase. It records:

- fresh games and positions, cumulative and per cycle;
- generation positions/second;
- search-versus-network policy KL, low-prior coverage, and immediate-win
  selection;
- optimizer examples consumed and replay consumption;
- historical/fresh presentation counts and active fresh archives;
- train policy/value/KL/illegal-mass metrics;
- validation metrics by historical/fresh source and opening/middle/late phase;
- validation-best diagnostic snapshot, latest actor hash, and monotonic global
  optimizer step.

## Strength evaluation

After successful completion, a dependent job compares the initial and final
checkpoints on 256 pre-generated opening pairs. Each opening is played with
colors reversed; rated search is noise-free. The arena rejects identical
hashes and reports Elo with uncertainty, but cannot alter the actor.

The run is operationally healthy only if all cycles and audits pass, metrics
stay finite, Adam steps remain monotonic, actors change, and the arena has zero
abnormal games. Playing strength is reported with its interval; it is never an
accept/reject gate.
