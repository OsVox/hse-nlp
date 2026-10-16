# Production input-pipeline batch probe

Compare batches 64, 128, 256 and 512 on the selected checkpoint, using the
same 32,768 original training-example presentations in two reversed rounds.
Use the actual encoded, two-batch-prefetch, length-grouped training input
path and BF16 AdamW updates. Warm-up updates and probe weights are discarded.
Synchronize once at the timing boundaries, rather than after every update.
CPU preparation checks that every size consumes equal examples and target
tokens, and records the model, tokenizer and training-pool hashes.

Choose a different batch only if both rounds finish and median examples per
second improve by at least 5% over 128. GPU activity and available memory
alone do not select a batch. A speed winner is not evidence that its learning
rate is suitable: compare a short equal-example continuation before changing
the full training recipe. Keep the original batch if quality deteriorates.

Prepared inputs remain in Drive. The probe must have a bounded launcher that
releases the runtime after saving results. No full training is started by
benchmark_production_batches.py. No measured speedup is claimed until this
probe has run on the intended training hardware.
