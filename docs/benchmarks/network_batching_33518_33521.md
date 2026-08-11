# GPU batching gate: jobs 33518--33521

The benchmark used the provisional standard 64-simulation soft-Z checkpoint
on one RTX 3070. Each setting ran for two seconds after an untimed warm-up and
used varied, legal 8x8 states.

| Batch | Latency (ms) | Leaves/second | Throughput vs batch 1 |
| ---: | ---: | ---: | ---: |
| 1 | 11.253 | 88.9 | 1.0x |
| 4 | 11.437 | 349.8 | 3.9x |
| 8 | 11.644 | 687.0 | 7.7x |
| 16 | 12.060 | 1326.7 | 14.9x |
| 32 | 12.908 | 2479.2 | 27.9x |
| 64 | 14.630 | 4374.5 | 49.2x |

The first three jobs failed before timing because the new scalar-versus-batch
test assumed decimal equality tighter than GPU convolution kernels provide:

- 33518 measured maximum policy drift `1.75e-5` between batch shapes.
- 33519 measured value drift `1.04e-4` on the `[-1, 1]` scale.
- 33520 measured a legal-policy sum of `0.9999999404` in float32.

Each failure was preserved and the job remained fail-closed. The final test
uses explicit bounds (`3e-5` absolute/`3e-4` relative for policy, `5e-4` for
value, and `1e-6` for normalization), while illegal logits must still receive
exactly zero probability. Job 33521 passed all 78 tests before benchmarking.

The scalar PUCT and lockstep multi-game coordinator also produce exactly equal
saved games under a deterministic batch evaluator, including root noise. The
small numeric difference is confined to GPU convolution batch shapes and does
not change the absolute-value or legal-policy contracts.
