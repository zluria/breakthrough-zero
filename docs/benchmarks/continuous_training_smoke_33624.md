# Continuous training smoke: job 33624

## Question

Can the rewritten learner publish the newest checkpoint as the actor, restore
the full Adam state, and continue training without any accept/reject state?

## Procedure

On one RTX 3070, the job used the preserved native 5x5 historical and neural
self-play archives. It trained two batches, checked `actor.json`, then loaded
`models.latest` and its TensorFlow checkpoint into a fresh process and trained
one more batch. It also ran `bash -n` on the three active generation wrappers.

This was a boundary test, not a strength experiment. The batches were 25%
historical and 75% fresh, with 128 examples per optimizer step.

## Result

- Slurm state: `COMPLETED`, exit `0:0`, elapsed 18 seconds.
- First process: optimizer steps 0 -> 2.
- Second process: strict restore at step 2, then step 2 -> 3.
- Both manifests named `latest` as the actor and `best_validation` as
  diagnostic-only.
- The actor hash changed after the resumed update.
- No candidate, champion, authorization, promotion, or rejection field was
  present.
- All active Slurm wrappers passed Bash syntax checking.

The smoke authorizes a measured native-mini learning run. It says nothing
about Elo, replay proportions, value targets, or exploration strength.
