# Short experiment results

Eight matched 1,000-update continuations from the same 70k checkpoint. All use greedy decoding and original training targets.

| Change | Pilot /245 | Confirmation /493 | Execution failures /738 | Public /960 |
|---|---:|---:|---:|---:|
| control | 137 | 279 | 40 | 562 (58.54%) |
| numeric2 | 134 | 283 | 42 | 562 (58.54%) |
| smooth005 | 135 | 290 | 44 | 555 (57.81%) |
| equal_question | 134 | 282 | 38 | 562 (58.54%) |
| dropout0 | 134 | 280 | 38 | 564 (58.75%) |
| dropout005 | 134 | 279 | 40 | 559 (58.23%) |
| lr6e5 | 135 | 284 | 40 | 559 (58.23%) |
| complete_targets | 138 | 285 | 45 | 555 (57.81%) |

No candidate passed the predefined gate: positive total gain, no negative split gain, and no increase in total execution failures. No second-seed confirmation, combination, or long run was launched. Filtering complete targets improved both subsets but raised total execution failures by five. Label smoothing improved total correct answers most, but regressed on the smaller subset and added four failures.

The best public checkpoint remains the earlier 2,000-update numeric-weighted model: 573/960 (59.69%). It is retained for reproduction, without treating public leaderboard selection as unbiased validation.

Compute: 998.35 seconds (16.64 minutes) from driver start through data restore, training, evaluation and saves; allocation and authorization before the driver are excluded. Median sampled GPU activity was 97%; mean was 63.44% across the full workflow including restore, saving and CPU evaluation. Colab Manage Sessions showed no active sessions after automatic release.

All eight checkpoints and 1,919-row exact greedy exports were saved to Drive, exported, hash/token/order verified and scored once by the bot. Receipts were saved back to the matching Drive checkpoint and read back byte-for-byte. No individual public failure example was used to create training targets or repair predictions. Reused validation limits the strength of all conclusions. T4 timing remains deferred.
