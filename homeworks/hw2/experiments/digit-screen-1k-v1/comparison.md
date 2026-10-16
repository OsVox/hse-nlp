# Individual-digit adaptation: rejected

Both arms continued our own 70k encoder-decoder Transformer for 1,024 updates, batch 128 (131,072 original example presentations), using identical optimizer state, seed, learning-rate schedule and strict greedy decoding. The treatment adds digit boundaries before BPE merges with no changes to the original vocabulary, token IDs or supplied target text. Longer sequences change length grouping and padding. Source: `f9676760c37607102647c7d0b792810f81f1e8f1`.

| Tokenizer | Pilot /245 | Confirmation /493 | Execution failures | Public /960 |
|---|---:|---:|---:|---:|
| Original BPE | 137 | 281 | 41 | 561 (58.44%) |
| Individual digits | 124 | 239 | 47 | 511 (53.23%) |

The treatment lost 55 held-out agreements and added 6 execution failures; it failed the predefined gate. Public score independently declined by 50/960 (5.21 percentage points). No extension or combination is justified. This rejects this short adaptation, not every possible model trained from scratch with digit tokenization. Held-out questions have been reused across experiments and remain exploratory.

CPU audit of 122,875 original pairs sampled across all 60 shards verified exact original-text roundtrips, unchanged vocabulary IDs, and checkpoint reloads. Mean question length rose 46.97→51.85 tokens and program length 74.52→80.94; at fixed 128/192 limits, question truncations rose 218→446 and program truncations 2,375→3,005. No synthetic examples entered training. Eighteen focused tests passed before launch.

All checkpoints and outputs were saved to Drive before export; all 1,919 programs were verified against saved token IDs, CSV/JSONL, question-order hash and model/source hashes. Each unique output was scored once by the course bot, and both receipts were uploaded and read back from Drive. The control has the same held-out and public correct counts as the previous same-setting control, but 11/1,919 generated strings differ; bitwise repeatability is not established.

CPU preparation took 149.47 seconds, including a 65.65-second tokenizer audit. GPU driver duration was 294.19 seconds (4.90 minutes), excluding allocation and Drive authorization. Sampled GPU utilization: median 97%, mean 58.18%, with restore, CPU scoring and saving included. Colab Manage Sessions confirmed no active sessions after automatic release. CPU use is approved; reuse unchanged verified artifacts to avoid unnecessary setup.

The best retained public model remains numeric2 at 573/960 (59.69%), rank 8/15. First place and T4 timing are still unfulfilled requirements. Its pinned inference notebook is unchanged.

Motivation: [Singh and Strouse, Tokenization counts (2024)](https://arxiv.org/abs/2402.14903) shows that number segmentation can affect arithmetic performance. It does not establish a benefit for this code-generation task, and no paper examples were used as training data.
