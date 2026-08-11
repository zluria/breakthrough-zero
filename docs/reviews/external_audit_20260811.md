# External audit received 2026-08-11

This report was supplied by an external software engineer. It is preserved
verbatim as an independent input to the project; implementation status is
tracked in the dated phase reviews and literature survey.

> The project is technically strong, but some choices have outrun the
> evidence. Before scaling, correct the following.
>
> **Native 5x5 network.** Replace padded 8x8 inputs/outputs with a proper 5x5
> CNN and 75-action head.
>
> **Use 5x5 as the tuning sandbox.** This is not merely a plumbing test.
> Literature-guided sweeps on 5x5 should establish sensible starting values
> for architecture, search, exploration, targets and training. Carry those
> choices to 8x8, then validate locally around them rather than restarting
> tuning from scratch.
>
> **Do a serious literature/code survey before more tuning.** Start with
> KataGo, Gumbel AlphaZero, resource-efficient AlphaZero work, Leela/KataGo
> implementation notes, and later search-control/sample-efficiency papers.
> For each relevant technique record: mechanism, claimed benefit, compute
> regime, evidence, complexity cost, and recommendation: adopt / test / reject.
>
> **Reconsider exploration.** “64 unique games” does not establish sufficient
> exploration. Literature should guide candidates: Dirichlet noise,
> Gumbel/sequential-halving approaches, temperature schedules, randomized
> starts/state archives, or combinations. Test the most promising simple
> alternatives on 5x5.
>
> **Reconsider value/policy targets.** Compare final outcome, soft-Z, simple
> mixtures, and literature-supported auxiliary/short-horizon targets. Preserve
> simplicity: extra heads are worthwhile only if evidence suggests meaningful
> sample-efficiency gains.
>
> **Tune search for limited compute.** On 5x5 study simulation budget,
> cPUCT/FPU, subtree reuse, batching and symmetry averaging. Comparisons must
> use equal wall-clock budgets. Gumbel-style search deserves particular
> attention because it targets low-simulation regimes.
>
> **Improve experimental discipline.** Tiny pilots identify disasters; they
> do not establish winners. Use 5x5 for broad cheap tuning, freeze a small set
> of choices, transfer them to 8x8, and perform narrower confirmation there.
> Important claims need fresh evaluation games and uncertainty, not repeated
> winner-picking from tiny arenas.
>
> **Independent adversarial review.** Builder self-review has already failed.
> Before expensive phases, use a fresh reviewer whose job is to identify the
> three highest-impact ways the proposed experiment could be misleading.
>
> Execution plan
>
> First fix native 5x5. Then conduct the literature/code survey and turn it
> into a short ranked list of candidate improvements. Run broad, cheap 5x5
> experiments informed by that survey. Select a coherent baseline and 2–3
> promising variants. Transfer the best 5x5 settings to 8x8 as initial
> parameters, perform small local retuning rather than a full search, generate
> pretraining data in stages, and evaluate learning curves before scaling
> further.
>
> Do not recreate 2018 AlphaZero by reflex. Use eight years of subsequent work
> to decide what deserves compute.

## Project qualification

The audit suggests studying FPU. This project has a stronger explicit
requirement: every unvisited child uses its parent's absolute Q. That invariant
is tested and will remain fixed unless the teacher changes it. All other audit
requests are accepted.
