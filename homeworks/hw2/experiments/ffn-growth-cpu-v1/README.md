# CPU check: grow our trained model without discarding learned weights

The active selected-scale-v1 training run is unchanged. This preparation does
not authorize or launch another GPU run, choose a new recipe, or establish an
accuracy improvement.

The hypothesis is that short scratch runs may not reveal the benefit of extra
capacity once a smaller model has learned the task. Rather than retrain all
weights, double only the gated feed-forward width from 1,536 to 3,072, keeping
the 512-wide hidden state, attention, tokenizer and 6/4 encoder/decoder depths.
The model grows from 42,481,152 to 66,074,112 parameters. It uses only our own
scratch-trained weights and randomly initialized new input projections.

Old input rows and output columns are copied exactly. New output columns are
zero, so the added units initially contribute zero. Their output weights get
gradients immediately; their input weights get gradients after the first
output-weight update. This follows the idea of function-preserving network
growth discussed in [Net2Net](https://research.google/pubs/net2net-accelerating-learning-via-knowledge-transfer/),
but is not its neuron-duplication algorithm. That paper's results are not
evidence of a quality gain on this task.

Four small CPU tests passed: logits and cached/uncached greedy outputs,
unchanged parent and global random state, learning in added units, and
rejection of non-growth widths. A separate real-checkpoint test passed on four
unchanged original training examples: zero maximum logit difference and equal
32-token greedy sequences with and without cache. The parent file hash was
unchanged. See verification.json for scope and runtime.

These tests do not establish equality on every question, identical dropout
noise during training, GPU speed, or improved accuracy. No grown checkpoint
has been trained, saved as a candidate, or submitted.

If the current continuation stops, a quality test would still need a fixed
time budget, a same-parent baseline, the same original training stream and
full sequences, matched optimizer handling, fixed held-out evaluation, and
Drive exports before bot scoring. Optimizer handling must be decided and
tested explicitly: resetting both arms loses mature moments; extending old
moments requires shape checks and a stated policy for the new coordinates.
Do not silently reset or reuse incompatible optimizer state. A successful
numerical growth check alone is not a reason for a long training run.
