# Actual T4 inference, 16 October 2026

Both immutable models were CPU-staged in Drive before GPU allocation. One
Tesla T4 ran FP32 batches 64, 128 and 256, then the reverse order. All twelve
runs generated 1,919 programs; all CSV/JSONL hashes and selected-token decodes
were verified locally after export from Drive.

| Model | Batch 64 median | Batch 128 median | Batch 256 median | Batch 256 slowest |
|---|---:|---:|---:|---:|
| Established 9.97M, step 180k | 28.885 s | 16.676 s | 10.586 s | 10.791 s |
| Larger 42.48M, step 6k | 37.032 s | 23.146 s | 20.304 s | 20.556 s |

Select batch 256 for these weights: 2.73x and 1.82x faster than batch 64 by
median time. Every run exactly matches its previously scored JSONL, preserving
53.85% and 18.96% respectively. No duplicate bot submission was made. The main
inference notebook now uses 256 for the established checkpoint. New weights
must be timed again; this does not certify future checkpoints.

The driver took 318.57 seconds including restoration of the two model bundles.
Reported generation timers include tokenization, sorting, generation, syntax
retry, decoding and local CSV writing. Full-call time and memory are retained
in each results.json; model loading and Drive archival are outside generation.
GPU release was requested automatically and completed manually when the UI
remained connected. CPU was selected for exporting the saved evidence.

The ZIP was saved in Drive before download; export.json records its SHA-256.
verification.json records the exact model/output identities and selected
settings. Full exact exports are in the Drive archive and the local Downloads
copy. The repository retains measurements and verification rather than twelve
duplicate copies of identical answer sets.
