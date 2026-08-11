# Root noise: purpose, placement, and experiments

## What problem noise solves

The policy controls which moves PUCT examines. Once a plausible move's policy
falls near zero, a small search may never revisit it; self-play then produces
more targets that omit it. Root noise is an intervention against that feedback
loop. It is not a source of value information and it is not intended to make a
rated agent stronger on one position.

Dirichlet noise changes a root prior once:

`search_prior = (1 - fraction) * network_prior + fraction * noise`

It is applied to legal root moves only, before simulations after root expansion.
It is never added at internal nodes, on every simulation, or to backed-up
values. A fresh sample is used at the next self-play root. Network and search
priors are stored separately.

AlphaZero used a noise fraction of 0.25 and scaled per-action concentration with
typical branching factor. KataGo describes the invariant more clearly as total
concentration about 10.8 divided among legal moves. We expose total
concentration, so a Breakthrough position with 10 moves and one with 25 moves
receive comparable-shaped noise.

Sources:

- [AlphaZero paper](https://arxiv.org/pdf/1712.01815)
- [AlphaZero.jl MCTS explanation](https://jonathan-laurent.github.io/AlphaZero.jl/v0.3/reference/mcts/)
- [KataGo shaped Dirichlet noise](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#shaped-dirichlet-noise)

KataGo explicitly warns that its shaped variant is suggestive rather than a
controlled measured improvement. Practitioner reports also repeatedly show
deterministic collapse and low-prior blind spots, but they are leads for our
diagnostics, not parameter evidence.

## Where we use it

- Rules/search tests: off.
- Uniform-policy rollout pretraining: off by default. Rollouts and stochastic
  move selection already diversify games; Dirichlet would add arbitrary policy
  bias before there is a learned blind spot.
- Neural self-play: opt-in and tuned.
- Rated search after the opening: off.
- Evaluation opening phase: Dirichlet off. The mini arena saves four uniform-
  random plies, exactly two moves per side.

The arena generates one candidate-independent random opening suite before
inspecting match results. Every matchup uses the same starts, and every start
is played twice with colors reversed. This is chess-style duplicate play:
randomness produces the saved starts, not unequal perturbations inside rated
agents. A separate robustness experiment may deliberately use search noise.

Four plies are specific to the 5x5 debug game. A six-ply PUCT-noise suite
already produced a trivial immediate-win opening in job 33478. Opening depth
must scale with game horizon; it is not a constant to copy between boards.

## Pilot, not a copied constant

For neural self-play, first measure Breakthrough's legal-move distribution and
policy entropy. Then run a fixed-network diversity pilot at equal wall time:

| Setting | Fraction | Total concentration |
| --- | ---: | ---: |
| Off | 0 | 10 |
| Moderate | 0.10 | 10 |
| AlphaZero-like mass | 0.25 | 10 |
| Sharper | 0.25 | 2.5 |

We log noised/raw KL, visit entropy, low-network-prior moves visited, unique
positions/openings, immediate-win misses, game length, and search throughput.
This rejects pathological settings but cannot prove learning benefit.

The expensive test is successive halving of at most three settings in short
end-to-end runs with equal wall time. Promising settings are repeated with more
seeds. Noise fraction and total concentration change separately whenever
possible. An early-only schedule is tested only if late-game diagnostics show
that noise consumes too much of the small search budget in tactical positions.

Move-selection temperature and root noise are distinct interventions. We do
not change both in one ablation.
