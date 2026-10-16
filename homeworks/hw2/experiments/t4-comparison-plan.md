# T4 comparison after the short training screen

Measure the established 9.97M step-180000 model and the 42.48M lower-rate
step-006000 finalist. CPU preparation verifies parent model hashes and the
official 1,919-row test file, loads the larger model for a short generation
check, then saves source and input bundles to Drive. Existing verified input
bundles for the established model can be reused. No GPU allocation before
exporting and verifying the preparation record.

For each immutable model, compare FP32 generation at batch 64, 128 and 256,
then reverse the order for the second round. Keep question order, token limits,
cached greedy generation and syntax-only beam retry unchanged. Compare every
program against batch 64; an inference batch change may change floating-point
results, so speed alone cannot establish retained quality. The existing
benchmark saves exact outputs, selected token IDs, hashes, per-call and
generation time, peak memory, load time and warm-up time. No profiling pass is
included in this two-model run, to keep its cost bounded.

Each model has a 600-second soft budget and a 720-second process timeout; the
notebook has an 1,800-second outer timeout. Test file transfer and model loading
are separate from timed generation. TensorFlow and Flax imports are disabled
for these PyTorch-only subprocesses. No model updates occur.

Save results to HSE_NLP_HW2/benchmarks/t4-comparison-v1, including an evidence
ZIP before requesting GPU release. Verify release in the UI and release
manually if necessary. Export from Drive, choose a consistently faster setting,
and send a chosen output to the bot if it differs from its already scored
export. A later full-training checkpoint still needs its own T4 timing.
