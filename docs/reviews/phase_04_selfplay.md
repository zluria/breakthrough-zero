# Phase 4 review: generate raw games without corrupting the evidence

## Where the algorithm stands

The rules, policy mapping, exact symmetries, absolute-value PUCT, dummy rollout
evaluator, and architecture-independent data schema are tested. What is still
missing is the small piece that repeatedly searches a position, selects a move,
and turns the resulting trajectory into durable game chunks.

Before the first pilot, we added a `5 x 5` one-row debug ruleset. It uses the
same padded bitboards, 192-action policy boundary, four symmetries, PUCT, and
storage schema as the target game. Ruleset identity is stored with every raw
position. Standalone move symmetries require that identity explicitly; they do
not guess an 8x8 default.

## Invariants this phase must protect

1. A recorded position is always non-terminal and its selected move is legal.
2. Applying each selected move produces the next recorded state exactly.
3. Applying the last selected move produces the saved absolute game outcome.
4. Search values are never negated. Player-to-move matters only inside PUCT
   selection.
5. A fixed seed reproduces states, searches, selected moves, and the result.
6. Root noise is opt-in. Dummy-MCTS pretraining starts without it so the first
   dataset has one less coupled variable.
7. Raw games are saved once in absolute coordinates. The four symmetries are
   derived later by the loader, not materialized four times on disk.
8. A chunk is usable only after both its data file and checksummed manifest are
   complete. The manifest is the final commit marker.

## Deliberate simplifications

- Each played position gets a fresh search tree. Reusing the selected subtree
  would save work, but it also mixes inherited visits with the new root budget
  and makes root-noise accounting easier to get wrong. Revisit this only after
  profiling a correct generator.
- Generation is sequential in the library. HPC parallelism will use independent
  processes and disjoint game seeds, which keeps failure recovery and
  reproducibility simple.
- Early moves sample from visit counts; later moves choose the most-visited
  action. Temperature is a move-selection rule, not a change to stored visits.

## Expected bottleneck

The random rollout value evaluator, not legal move generation or chunk writing,
should dominate dummy pretraining. The first timing run must report games,
positions, searches, and positions per second. We should optimize only after
that profile, and keep the uniform rollout as the reference implementation.

## Smallest justified next step

Implement one deterministic game generator, chronological record validation,
atomic chunk publication, and a tiny command-line driver. Test terminal handling,
reproducibility, visit sampling, interrupted writes, and round trips before any
bulk generation or GPU allocation.

## Result of the local gate

The gate passed on 2026-08-11:

- 46 unit tests pass, including optimized-versus-literal legal moves on random
  complete games for both board sizes.
- A two-game mini pilot wrote 31 positions in two immutable chunks. A second
  invocation verified both manifests and seeds and generated nothing.
- A same-seed six-game timing check reached 45.5 positions/s with uniform
  rollouts and 42.7 with tactical rollouts. The sample is too small for a
  strength or throughput conclusion, so uniform remains the reference and the
  tactical preference remains opt-in.

The next justified action is a bounded HPC environment smoke test, not bulk
self-play: verify the checked-in code, Python environment, GPU visibility, and
one Keras forward/train/save/load cycle on an idle RTX3070 node.
