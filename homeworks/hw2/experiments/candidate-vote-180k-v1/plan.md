# Four-candidate execution agreement pilot

Status: prepared, no quality result yet. No change to the running 42M training.

Hypothesis: selecting an unchanged model program from a strict majority of
numeric execution results improves answer accuracy beyond choosing the top
beam alone. This is an inference-only experiment; it creates no targets or
training data and never uses reference answers to choose a candidate.

Use the fixed 9.97M, step-180000 model (SHA256
0399a924eea6ef872aa0b403b77fc5e33d24909cc7690536880bc00b254362c7),
FP32, four deterministic beams, four returned programs, batch 32, input/output
limits 128/192. Compare top beam, first executable beam, and strict majority
of all four candidates. Ties retain model rank. The baseline here has no
syntax retry, unlike the established full-test inference path.

First use reserved training rows 512:768. Proceed to a confirmation on 768:1024
only if a policy improves numeric correct count over top beam. Require a
positive gain on both subsets before considering full-test generation; this
is an operational gate, not statistical significance. These held-out question
texts are excluded from the current training stream. No public test labels or
bot failure examples enter the experiment. The historical 180k training lineage
has the documented earlier validation-copy caveat.

The current G4 host has 48 CPU cores. Run this pilot with CUDA hidden, at most
four CPU threads and affinity to the last four cores, while the main training
continues on GPU. Bound the child process group to 360 seconds, save output
and logs to Drive, and never release the training runtime from this side job.
Check training throughput before/after; stop the pilot if it harms training.
Export results from Drive and retain exact candidate/selected token IDs.

The inspiration is answer agreement from Wang et al.,
[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171).
Our deterministic beam-candidate experiment differs from their sampling method;
its outcome must be measured. Restricted execution is a diagnostic, not a
replica of the bot sandbox. T4 performance remains deferred.
