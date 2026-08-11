# Neural replay generation 1

This directory records the manifests for the first bounded neural self-play
batch. The four raw schema-3 NPZ chunks are preserved in the GitHub release
[`neural-replay-r01`](https://github.com/zluria/breakthrough-zero/releases/tag/neural-replay-r01)
and on the HPC under:

- `/home/zurlu/breakthrough-zero-data/neural-selfplay-r01a-33534`
- `/home/zurlu/breakthrough-zero-data/neural-selfplay-r01b-33535`

The batch contains 256 standard-board games and 18,110 positions. Loading a
chunk verifies the SHA-256 digest in its matching manifest before rebuilding
and validating every game trajectory.
