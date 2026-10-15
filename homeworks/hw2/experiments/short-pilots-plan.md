# Short pilots before a full training run

Requested on 2026-10-15. This is a predeclared comparison, not measured results.

All four runs start from random weights, use the same seed and data stream,
128 examples per update, mixed samples from all 60 original shards, the same
8,192-token tokenizer, and 128/192 token limits. No new training targets.
First stage: 2,000 updates, 200 warm-up updates, cosine schedule, selection loss
every 500 updates, one immutable Drive checkpoint at 2,000. Models: 9.97M with
4/4 layers at learning rate .001; 35.66M with width 512 and 4/4 layers at .001;
42.48M with width 512 and 6/4 layers at .001 and .0003. Keep dropout .1.

This separates width, extra encoder depth, and a lower learning rate. Decoder
depth is four throughout. Compare both quality at matched examples and the
elapsed GPU time including checkpoint generation/export. Earlier throughput
measurements do not establish the best batch size for these larger models.
Use the same batch initially for a fair screen; inspect memory and utilization
before tuning the selected model's batch size.

Use reserved training rows 0–255 for token loss and rows 256–511 for program
execution diagnostics; exclude all 1,024 held-out question texts from training.
Rows 512–1023 are reserved for a later audit. Treat the small numeric diagnostic
as noisy; retain two promising configurations, including a larger candidate if
its loss trajectory is still improving despite slower early task accuracy.
Extend those two to 6,000 updates and repeat the comparison before a full run.
The short cosine schedules and one seed do not determine final convergence.
A second-seed check is appropriate if finalists remain close.

Save every checkpoint, exact test prediction, token provenance, metrics and
logs to Drive. Export each checkpoint's 1,919 predictions from Drive and submit
once to the course bot, keeping each receipt. Bot results confirm progress;
do not turn test feedback into training targets. Measure finalist inference on
an actual T4 before starting a long run. The first screen has an 1,800-second
soft budget and 900-second timeout per trial; notebook execution has a separate
3,000-second process timeout and releases compute in finally. No automatic
full training launch is included.
