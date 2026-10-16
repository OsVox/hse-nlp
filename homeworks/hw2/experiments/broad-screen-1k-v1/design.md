Eight short matched continuations from the verified 70k checkpoint. Each has 1,000 updates, batch 128, unchanged architecture and greedy decoding. Only the named factor changes. Filtering discards overlong original targets and refills batches; it changes the data stream by design.

Label smoothing and dropout are motivated by section 5.4 of https://arxiv.org/html/1706.03762v7. Their benefit for this model is a hypothesis, not an established result. Numeric weighting and equal-question loss are task-specific hypotheses.

Rank held-out executed-answer agreement, inspect execution errors, and repeat up to three survivors with a second seed. Test combinations in a short matched run before considering a longer run. Public bot scores are retained separately; never inspect individual public-test failure examples to construct training changes. Validation has been reused and is exploratory.

Every final checkpoint, original greedy output, source identity and telemetry are saved to Drive. GPU is released automatically on completion or timeout.
