# CPU audit: Python tokenizers and original target consistency

Completed 2026-10-16. Source: `52cccd19da35f70d023cf63f2aeb0d161524de18`. Driver: 201.29 seconds, CPU only; successful exit and runtime release verified. No model training or bot submission occurred in this audit. The best checkpoint remains unchanged at 573/960 (59.69%), last verified rank 8/15.

## Tokenizers requested by the user

Fit on 61,440 original question/program pairs drawn across all 60 course training shards. Evaluate on 15,304 distinct question texts, excluding every exact copy of a fitting or held-out question. All four new tokenizers use the same fitting and evaluation rows. The existing tokenizer was previously trained on shard 0; this comparison measures compression, not unbiased generalization against that reference.

| Tokenizer | Mean question tokens | Mean code tokens | Code change vs current | Programs over 192 tokens |
|---|---:|---:|---:|---:|
| Current BPE, 8k | 46.86 | 74.11 | — | 278 |
| Fresh balanced BPE, 8k | 46.50 | 73.30 | -1.08% | 250 |
| Code-weighted BPE, 8k | 46.98 | 73.09 | -1.36% | 250 |
| Explicit Python boundaries, 8k | 83.65 | 86.21 | +16.34% | 556 |
| Code-weighted BPE, 16k | 45.44 | 71.76 | -3.17% | 219 |

All five tokenizers recover every evaluated question and program exactly, with no unknown tokens. This includes significant Python whitespace. The code-weighted variants repeat unchanged original program strings three times per question during tokenizer fitting; no extra examples or targets are introduced. Explicit Python boundaries split identifiers, numeric strings, operators and whitespace before BPE. They reduce mean pieces per numeric span from 1.1594 to 1.0309 in code, but greatly increase overall sequence lengths and truncation. Reject this particular shared-tokenizer boundary rule; this does not reject every possible Python-aware tokenizer.

Code emphasis alone has a small effect: code8192 is only 0.28% shorter than the fresh balanced8192 control. The 16k vocabulary adds 4,194,304 parameters to the shared embedding table at width 512, increasing the 42,481,152-parameter architecture to 46,675,456. A larger output vocabulary also changes generation cost; shorter sequences do not prove faster execution or better accuracy.

Keep code8192 and code16384 as a matched vocabulary-size pair for a later short training comparison. No vocabulary has been inserted into the trained best checkpoint, and no score improvement is claimed. Any reuse of existing weights would need explicit token remapping and its own validation; a direct vocabulary replacement is invalid.

Methodology reference: [Hugging Face CodeParrot](https://github.com/huggingface/blog/blob/main/codeparrot.md), which motivates fitting a code-specific tokenizer. Its external corpus and pretrained tokenizers were not used. The exact sample positions, original file hashes, tokenizer hashes, numeric-fragmentation diagnostics and measured timings are in `tokenizers/`.

## Original-program consistency

Independently sample exact question groups with a fixed keyed hash, excluding every held-out question copy. Across 11,836,444 eligible original rows, the sample contains 11,460 rows in 10,510 question groups. Of 701 repeated groups, 198 have identical program strings and 503 have different programs. Execute 1,126 distinct original programs from those 503 groups; five groups reach the eight-program cap.

| Diagnostic category | Groups |
|---|---:|
| All tested programs execute and agree | 284 |
| At least two executable programs disagree | 177 |
| Fewer than two executable programs | 38 |
| Executable programs agree, with unresolved programs | 4 |

The 177/503 (35.19%) conflict rate applies to sampled repeated questions with different program strings, not to all training rows. A conflict does not identify the correct program. Agreement does not prove correctness. Unsupported execution remains unresolved; it is not evidence that a target should be deleted. This diagnostic uses the existing restricted executor, not the course bot's sandbox.

The result supports investigating an original-data filtering pilot. No filtering was applied, no correct answers were selected, and no execution values were converted into training targets. Saved evidence contains question/program hashes, original shard/row positions and statuses. Five focused tests verify grouping, held-out-copy exclusion, unchanged source files, classification, disjoint tokenizer fitting/evaluation and serialized round trips.

## Saved evidence

[Drive experiment folder](https://drive.google.com/drive/folders/1kooaf-IGvDSNEce_jvQE8Bf2gQVECHPg). Source and launcher were uploaded and read back before CPU allocation. Data archive and per-file hashes were checked on restore. All tokenizer and audit artifacts were saved directly to Drive, exported and hash-verified. `execution.complete.json` records return code zero; `runtime-release.json` and `compute-released.png` record no active sessions. T4 inference timing remains deferred and unverified.
