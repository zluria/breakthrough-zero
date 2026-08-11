# Phase 22 review: fixed first-generation training

## Question

Does one small update on neural self-play improve the selected pretrained
checkpoint, and does final outcome or soft-Z provide the safer value target?
This is the common regression point in student projects, so the fixed data is
more valuable than immediately generating another batch.

## Controlled comparison

Train two models from the same selected 64x4 checkpoint and the same two
128-game replay shards. Use the same whole-game 80/20 split, sample order,
random augmentation stream, batch size, learning rate, and epoch count. Change
only the absolute value target:

- `outcome`: the final absolute Player-1 game result;
- `soft_z`: the absolute Player-1 root mean Q retained in the raw record.

The policy target is the same root visit distribution in both runs. Raw data
remains unaugmented; the loader chooses one of the four exact symmetries on
each training draw. Validation uses the identity transform for a stable metric
and is split by game, not position.

Use Adam at `3e-4`, batch size 128, and five epochs. This is a conservative
fine-tuning rate and roughly 600 updates for the expected data volume. Do not
resume optimizer state from pretraining: the new learner records a clean
optimizer boundary while retaining network weights.

## Required implementation change

The training script currently builds only a fresh model and accepts one input
root. Add two explicit, readable options:

- `--initial-model PATH` loads weights/architecture and verifies they match the
  stated architecture flags;
- repeatable `--extra-input PATH` adds immutable chunks without copying them.

Reject duplicate chunk paths and duplicate recorded game seeds so a shard
cannot silently receive extra weight or leak between the train and validation
sets. Record every chunk path and checksum plus the initial checkpoint hash.

## Stop conditions

Do not train unless both self-play jobs pass tests, checksum reload, terminal
validation, and configuration audit. Do not call lower validation loss a
playing-strength win. The next gate is a duplicate-opening, color-reversed
arena under equal wall-clock move budgets, including the unchanged pretrained
checkpoint as a control.
