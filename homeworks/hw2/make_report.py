"""Build the HW2 report from retained experiment evidence."""
from pathlib import Path
import json
from html import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).parent
OUTPUT = ROOT / 'report.pdf'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body', fontName='Helvetica', fontSize=10, leading=14, spaceAfter=9))
styles.add(ParagraphStyle(name='Head', fontName='Helvetica-Bold', fontSize=13, leading=17, spaceBefore=10, spaceAfter=7, textColor=colors.HexColor('#243e59')))
styles.add(ParagraphStyle(name='SmallText', fontName='Helvetica', fontSize=8, leading=11, spaceAfter=6))
styles['Title'].fontName = 'Helvetica-Bold'
styles['Title'].fontSize = 23
styles['Title'].leading = 28


def p(text, kind='Body'):
    return Paragraph(text, styles[kind])


def table(rows, widths):
    result = Table([[p(escape(str(x)), 'SmallText') for x in row] for row in rows], colWidths=widths, repeatRows=1)
    result.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f7')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,0),0.5,colors.HexColor('#8fa7be')),
        ('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#dbe3ec')),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    return result


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#637487'))
    canvas.drawString(20*mm,13*mm,'HSE NLP HW2 | Andrei Vorsin | 16 October 2026')
    canvas.drawRightString(190*mm,13*mm,str(doc.page))
    canvas.restoreState()


def build():
    metrics = json.loads((ROOT/'metrics.json').read_text())
    submission = json.loads((ROOT/'submissions/2026-10-15-gpu-check.json').read_text())
    quality_path = ROOT/'quality_results.json'
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
    current = json.loads((ROOT/'experiments/mixed-lr001-20k-drive-v1/summary.json').read_text())
    continuation = json.loads((ROOT/'experiments/continue-20k-60k-b128-v1/summary.json').read_text())
    extended_root = ROOT/'experiments/continue-60k-180k-b128-v1'
    extended = sorted([json.loads(path.read_text()) for path in
                       (ROOT/'submissions').glob('2026-10-15-extend-b128-step-*.json')],
                      key=lambda item: item['step'])
    best = max(extended, key=lambda item: item['public_percent'])
    selection = json.loads((extended_root/'selection.json').read_text())
    speeds = selection['median_examples_per_second']
    gain = 100*(speeds[selection['chosen']]/speeds['legacy']-1)
    completion_path = extended_root/'execution.complete.json'
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
    t4_path = ROOT/'experiments/inference-speed-180k-t4-v1/results.json'
    if (ROOT/'experiments/t4-comparison-v1/baseline180k/results.json').exists():
        t4_path = ROOT/'experiments/t4-comparison-v1/baseline180k/results.json'
    t4 = json.loads(t4_path.read_text()) if t4_path.exists() else None
    large_t4_path = ROOT/'experiments/t4-comparison-v1/encoder6-6k/results.json'
    large_t4 = json.loads(large_t4_path.read_text()) if large_t4_path.exists() else None
    latest_best = json.loads((ROOT/'experiments/current-best.json').read_text())
    latest_rank = latest_best.get('latest_leaderboard_observation', {})
    story = [p('Code generation from scratch','Title'),
        p('HW2 experiment report - work in progress<br/>Andrei Vorsin | 16 October 2026','SmallText'),
        p('Current result','Head'),
        p(f"Verified and scored checkpoints reach {extended[-1]['step']:,} updates. Ten initial checkpoints, eight first-continuation checkpoints and {len(extended)} longer-continuation checkpoints were saved in Google Drive and scored by the course bot. The best public result is <b>{latest_best['public_percent']:.2f}% ({latest_best['public_correct']}/960)</b>. Latest verified rank is {latest_rank.get('rank', latest_best['reported_rank'])} of {latest_rank.get('participants', latest_best['participants'])} at {escape(latest_rank.get('observed_at', latest_best['timestamp']))}. Its exact scored output is {escape(latest_best['filename'])}. This is the current best result across our decoding experiments and uses strict greedy decoding. Later sections distinguish historical settings. T4 timing remains deferred. No private result is available."),
        p('Task and rules','Head'),
        p('Train an encoder-decoder Transformer from random weights using only the supplied SlavaYus/tinysms_qa_12m training split. Generate one Python program per test question. All 1,919 programs must be generated within 180 seconds on one NVIDIA T4. No pretrained weights, outside examples, model distillation, or agent-written target programs are used.'),
        p('Data and architecture','Head'),
        p('The allowed dataset contains 11,838,003 question/code pairs in 60 training files. The test file contains questions only. A byte-level BPE tokenizer with 8,192 tokens was fitted on shard 0 after reserving its first 1,024 rows. The initial baseline has 9,967,360 parameters: four encoder layers, four decoder layers, width 256, feed-forward width 768, four attention heads, gated GELU and dropout 0.1. It uses the T5 architecture with randomly initialized weights. The current selected larger model has 42,481,152 parameters, six encoder layers, four decoder layers, width 512 and feed-forward width 1536; the full run stopped at 70k, the matched loss trials each added 2k updates, and the current best adds 1,024 control updates.'),
        p('Training uses AdamW, 500 warmup updates, cosine decay to 10% of the initial rate, weight decay 0.01 and gradient clipping at 1.0. The first 20,000 updates used batch 64 and peak learning rate 0.001. The first continuation retained optimizer moments, used batch 128 and restarted the schedule at peak rate 0.0005. The 60k-to-180k continuation retains batch 128 and optimizer moments, with a new schedule peaking at 0.0003. Input and target limits are 128 and 192 tokens. Padding tokens do not contribute to the loss. Model and optimizer states are saved regularly.'),
        p('Validation correction','Head'),
        p('Earlier runs excluded the first 1,024 row positions but did not exclude copies of their questions. An audit found 24 later rows in shard 0 with the same question text. Historical validation was therefore not fully independent. Future streams exclude every reserved question text across all shards. Rows 0-255 select checkpoints; rows 256-511 provide separate diagnostics. The earlier tokenizer may have seen duplicate question text.'),
        PageBreak(),
        p('Quality experiments','Title'),
        p('Historical runs','Head'),
        table([
            ['Experiment','Evidence','Interpretation'],
            ['Local MPS / short T4 checks','One local update; ten T4 updates; later T4 pilot reached 4,000 updates, loss 1.6469.','Pipeline checks. The longer pilot checkpoint was not recovered.'],
            ['Sequential shards, 20,000 updates','Final held-out token loss 1.1213; best logged 0.9207 at step 12,500.','Best weights were not retained; last weights only.'],
            ['Rotate after 1,000 mini-batches, 20,000 updates','Final loss 0.9815; saved best 0.8636 at step 18,500.','Selected historical checkpoint. Held-out exact code match 2/256; public score 18.85%.'],
        ],[42*mm,68*mm,60*mm]),
        p('The historical comparison supports broader data coverage by token loss, but the validation-copy issue limits that conclusion. Token loss scores the next token with the reference prefix supplied; it does not show whether a generated program computes the right answer.'),
        p('Audit before the new pilots','Head'),
        p('All 1,024 held-out programs round-trip through the tokenizer exactly with cleanup disabled. Only 2 held-out questions and 3 of 1,919 test questions exceed the input limit; 5 held-out programs exceed the target limit. Truncation and tokenizer corruption do not explain most failures in this audit. Samples of 1,000 rows from shards 0, 15, 30, 45 and 55 have mean code lengths 65.96, 62.56, 93.17, 61.84 and 118.95 tokens. This suggests that sequential training encounters different styles for extended periods.'),
        p('Matched sampling pilot','Head'),
        p('Hypothesis: mixing original examples across shards improves held-out numeric agreement. Two fresh runs use identical initialization, model, seed, BF16 precision, batch 64, accumulation 1, grouping window 32 and 6,000 updates. Both exclude validation question texts. The control rotates after 1,000 batches. The treatment mixes up to 4,096 randomly selected original rows from every shard and resamples each epoch. Sampling is the experimental variable. Comparisons with older runs also change batch shape and validation exclusion.'),
    ]
    records = quality.get('runs',{})
    if records:
        rows = [['Run','Best token loss','Held-out numeric agreement']]
        for name, item in records.items():
            val = item.get('best_validation',{})
            num = item.get('numeric',{})
            score = f"{num.get('numeric_correct','?')}/{num.get('usable_references','?')}"
            rows.append([name, f"{val['token_loss']:.4f}" if 'token_loss' in val else 'not measured', score])
        story += [table(rows,[55*mm,50*mm,65*mm])]
    else:
        story += [p('The matched pilot was still running when this report version was generated. No pilot improvement is claimed.')]
    story += [p('Numeric evaluation compares solve() outputs of generated and supplied held-out programs, using limited separate processes and restricted imports. Only finite numeric reference results are usable. The old saved model agrees on 50/245 usable references (20.4%). Reference errors and runner restrictions can affect this measure. Numeric outputs are never used as training targets.'),
        PageBreak(),
        p('Completed Drive-backed run','Title'),
        p('All ten checkpoints retained','Head'),
        p('The mixed sampler completed 20,000 updates with batch 64, BF16 training and a peak learning rate of 0.001. Source commit: 1a23e45b1138c3dba7c40befeae6d3337153df98. Training plus periodic validation, generation and Drive saves took about 12 minutes on an NVIDIA RTX PRO 6000 Blackwell Server Edition. This is not a T4 timing measurement.'),
        table([['Step', 'Validation token loss', 'Public correct / 960', 'Public score']] + [[f"{x['step']:,}", f"{x['token_loss']:.4f}", x['correct'], f"{x['public_pass_at_1_percent']:.2f}%"] for x in current['checkpoints']], [35*mm, 50*mm, 45*mm, 40*mm]),
        p('Every checkpoint was written to Drive with optimizer state and a completion manifest before predictions were generated. Each checkpoint has 1,919 exact model outputs, matching CSV and JSONL files, validation metrics and provenance hashes. The files were exported from Drive, checked locally and then submitted to the bot. Colab was disconnected before the bot evaluations finished.'),
        p('Drive location: MyDrive/HSE_NLP_HW2/runs/mixed-lr001-20k-drive-v1. The repository keeps exported prediction evidence and individual submission receipts. Weights stay in Drive. The final validation loss is lower than in the short pilots, but scores across checkpoints show that lower token loss does not guarantee higher program accuracy.'),
        PageBreak(),
        p('Longer training and GPU use','Title'),
        p('Eight more checkpoints, all scored','Head'),
        p('The continuation added 40,000 updates and 5,120,000 example presentations in 1,957.2 seconds (32.6 minutes), including validation, generation and Drive saves. It resumed our own step-20,000 model. Source commit: a411a64f8afe61e3a2b337af338c0ea6c19c9c15. No new target programs or external model weights were introduced.'),
        table([['Step', 'Validation token loss', 'Public correct / 960', 'Public score']] + [[f"{x['step']:,}", f"{x['token_loss']:.4f}", x['correct'], f"{x['public_pass_at_1_percent']:.2f}%"] for x in continuation['checkpoints']], [35*mm, 50*mm, 45*mm, 40*mm]),
        p('Batch size was chosen from two matched 16,384-example probes with reversed order. Median throughput was 2,100 examples/s at batch 64; 2,279 at 128; 2,114 at 256; and 1,950 at 512. Batch 128 was 8.5% faster than 64, with peak allocated memory 7.75 GiB. Larger batches raised utilization but reduced useful throughput.'),
        p('Across the continuation, 1,972 one-second GPU samples averaged 60.38% utilization, with median 78%, 16.23% zero samples and peak used memory 10,397 MiB. These include checkpoint saving and generation. CPU preparation occupied about 41% of the batch-128 probe, suggesting a remaining bottleneck. The GPU was released and Colab showed no active sessions at 19:07 +03:00.'),
        p('Drive location: MyDrive/HSE_NLP_HW2/runs/continue-20k-60k-b128-v1. Every checkpoint has exact 1,919-row outputs in official test order, provenance hashes and a bot receipt. The step-60,000 Blackwell generation took 9.07 seconds; its T4 timing remains unverified.'),
        PageBreak(),
        p('Faster input preparation','Title'),
        p('Matched CPU and GPU checks','Head'),
        p('CPU preparation compared the original batch path, reuse of window tokenization, and a bounded two-batch preparation queue. All three produced identical input IDs, masks and labels for 32,768 examples and 2,419,826 target tokens. Preparation took 18.72, 6.38 and 5.98 seconds respectively. Eighteen local checks covered truncation, duplicates, partial windows, literal special tokens and producer failures. Synthetic fixtures stayed confined to tests.'),
        table([['Input path','Median examples / second']] + [[name,f'{speed:,.2f}'] for name,speed in speeds.items()],[105*mm,65*mm]),
        p(f'Two GPU rounds used reversed variant order, the same checkpoint, seed, examples, precision and batch size. The selected two-batch queue improved median throughput by {gain:.2f}% over the original path, with peak allocated tensor memory 7.75 GiB. The tokenizer used by the producer is separate from the validation tokenizer. This is a matched speed result; utilization alone does not measure useful throughput.'),
        p('Longer continuation: 60k to 180k','Head'),
        p('The run resumes our own verified 60,000-step checkpoint and schedules 120,000 additional updates (15,360,000 example presentations). It keeps the mixed sampler, batch 128 and BF16 training. Source commit: ca26869455692cb570b94cb4a633be7701221f9e. Weights and exact predictions are saved to Drive every 20,000 updates before export and bot evaluation.'),
        table([['Step','Validation token loss','Public correct / 960','Public score']] + [[f"{x['step']:,}",f"{x['validation_token_loss']:.4f}",x['correct'],f"{x['public_percent']:.2f}%"] for x in extended],[35*mm,50*mm,45*mm,40*mm]),
        p('Completion has not yet been verified for this continuation; later scheduled checkpoints and GPU release remain pending.' if completion is None else f"The saved completion record reports return code {completion['returncode']} and {completion['checkpoints']} checkpoints. All six were scored. The last 20,000 updates added one public correct answer (516 to 517). Training was stopped at its planned endpoint, GPU release was verified, and subsequent file work ran on CPU."),
        p('Each export is checked against its Drive manifest, source commit, checkpoint hash, official question-order hash and exact decoded selected token IDs. Bot receipts are written and read back on Drive. The current-generation measurements use Blackwell and do not establish T4 compliance.'),
        PageBreak(),
        p('Generation and speed','Title'),
        p('Measured optimization experiments','Head'),
        p('The course materials motivated length grouping, mixed precision, gradient accumulation, clipping and caching. The model already uses cached autoregressive generation. Training disables that cache. Loss accumulation stays on the GPU until logging. These choices were measured rather than assumed to make the full workload faster.'),
        table([
            ['Blackwell inference, same weights','Median seconds','Changed outputs'],
            ['Before, FP32','11.350','0'],
            ['After, FP32','11.240','0'],
            ['After, FP16','13.027','54'],
            ['After, BF16','12.249','313'],
        ],[95*mm,37*mm,38*mm]),
        p('Two repetitions used reversed variant order on the second pass. The approximately 1% FP32 difference has overlapping timing ranges; it is not convincing evidence of a speedup. Lower inference precision was slower and changed programs. FP32 remains the default. An 80-update continuation benchmark also found no clear speed or validation-loss improvement; it does not establish full retraining quality.'),
        p('T4 requirement','Head'),
        p('The historical 39.09-second T4 measurement belongs to an older run. Blackwell timings cannot establish the T4 requirement for the current candidate. A bounded final-checkpoint T4 experiment is reported separately below; only its measured configurations can establish compliance.'),
        p('Exact model output and submission','Head'),
        p('Generation sorts questions by token length and restores their original positions. Historical runs followed greedy decoding with four-beam generation for syntactically invalid outputs. As of 16 October, strict greedy is selected: one candidate, no sampling, no syntax fallback, and no execution voting. No candidate is rewritten. Selected token IDs are saved, then decoded again to check equality with every output string. A sidecar records source commit, checkpoint hash, question-order hash, output hash, device and generation settings. JSONL conversion only wraps each exact string as a code field.'),
        p('The historical submission included one handwritten identifier repair, which does not meet the user\'s stricter model-only condition. That repair has been removed from all future inference. The old score is retained as historical evidence, not a score for a newly verified model-only candidate.'),
        p('The user confirmed that the required input is the official test.parquet question column in its original 1,919-row order. Generation preserves that order. The separate pass1_questions.jsonl is no longer a prerequisite.'),
        PageBreak(),
        p('Agent use and reproducibility','Title'),
        p('What the agent did','Head'),
        p('Codex read the assignment and course weeks, wrote data access, training, inference and diagnostic code, controlled Colab through MCP, inspected GPU state, compared measurements, prepared Telegram file packaging and wrote this report. It identified incomplete validation exclusion, stale T4 claims and handwritten postprocessing. Those findings changed the data exclusion rules, provenance records and speed checks.'),
        p('Codex did not provide solution programs as training targets or replace model answers with its own solutions. No external model was used as a teacher. The submitted programs must come from the trained Transformer. Bot examples are not used to tune individual answers. The humanizer skill is used only to make report prose direct; it does not conceal agent involvement.'),
        p('What remains open','Head'),
        p('The early pilot weights were lost when temporary runtime storage was released before export succeeded; their measurements remain saved. The revised workflow passed CPU save and restore checks and retained all ten initial, eight first-continuation and six longer-continuation checkpoints, exact predictions and bot receipts in Drive. The training GPU was released. A child-process release request failed at the notebook disconnect step; the new inference notebook executes its final release from the notebook kernel and has a hard process timeout.'),
        p('Longer training exceeded the 33% public baseline, but the first-place objective remains unmet. The best candidate and all intermediate checkpoints have actual bot scores. Changed decoding outputs require a new quality comparison; speed alone does not establish correctness. A public first-place result would not establish private first place. No private result is claimed.'),
        p('Reproduction files','Head'),
        p('Repository: <link href="https://github.com/OsVox/hse-nlp/tree/2026/homeworks/hw2">OsVox/hse-nlp, branch 2026, homeworks/hw2</link>. The directory contains train.py, data.py, numeric_eval.py, inference.py, tokenizer files, saved model weights, inference and training notebooks, experiment records and submission receipts. quality_experiments.md specifies the controlled pilot before its run. The raw logs and run configurations accompany the quality results.'),
        p('Use a fresh output directory for each experiment. Run training with the recorded settings and keep latest plus best checkpoints. Run numeric evaluation on reserved training rows only. Use hw2_inference_colab.ipynb on a T4 to generate and package the test outputs. Verify every checkpoint in Drive before releasing compute; export predictions and record bot receipts from Drive.'),
        p('Sources','Head'),
        p('<link href="https://github.com/ashaba1in/hse-nlp/blob/2026/homeworks/hw2_code_generation.md">Course assignment</link>; <link href="https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m">allowed dataset and schema</link>; course week 2 (encoder-decoder attention and decoding), week 3 (training methods), week 4 (attention and caching). Local measurements: metrics.json, gpu_benchmark/*.json, quality_results.json when present, and submissions/2026-10-15-gpu-check.json.','SmallText'),
    ]
    story += [PageBreak(),p('Alternatives to more updates','Title'),
        p('Held-out decoding and checkpoint averaging','Head'),
        p('Both experiments used reserved training questions, with no public answers or bot failure examples used for selection. The numeric runner executes supplied references and exact model outputs with time and resource limits; 245 of 256 references were usable. Agreement is a diagnostic, not the course bot score.'),
        table([['Experiment','Numeric agreement','Generation seconds, CPU','Decision'],
            ['120k greedy','120/245','10.512','Retain'],
            ['120k four beams','125/245','57.190','5.44 times slower; not adopted'],
            ['140k unchanged','122/245','10.160','Retain'],
            ['Average of 120k + 140k','124/245','10.313','Mixed evidence; not adopted']],
            [52*mm,34*mm,40*mm,44*mm]),
        p('Averaging used identical architecture, tokenizer and training-run settings. Parent weights were verified and averaged in CPU float64, then cast back and reopened. It adds no training targets or extra inference model. Validation token loss worsened from 0.452825 to 0.459383, and execution failures rose from 16 to 22. The two-answer diagnostic gain does not justify replacing the parent.'),
        p('Scaling-law estimate','Head'),
        p('The existing train-only probe contains 15,358 rows, sampled about equally across the 60 shards. With our tokenizer and length limits, mean question length is 46.91 tokens and mean code length is 73.83. Extrapolation gives approximately 555 million source tokens and 874 million target tokens per corpus. The last shard is smaller, so these are rough, non-deduplicated estimates. About 0.15% of sampled questions and 1.94% of sampled programs exceed the length limits.'),
        p('Chinchilla suggests about 20 training tokens per parameter as a rough compute-efficient guide. Dividing target tokens by 20 gives 44 million parameters; counting source plus target gives 71 million. These are two accounting proxies, not a confidence interval or a fitted optimum for encoder-decoder code generation. Encoder and decoder capacity have different scaling behavior, and lower token loss need not imply higher task accuracy.'),
        p('Proposed first larger model: width 512, feed-forward width 1536, six encoder layers and four decoder layers, totaling 42,481,152 parameters. Keeping four decoder layers limits repeated work during generation. At unchanged efficiency and time proportional to parameter count, one corpus-sized amount of data would take roughly 3.8-4.4 hours on the measured training GPU. This is a planning estimate; a short throughput/quality pilot and actual T4 timing must precede long training.'),
        p('<link href="https://arxiv.org/abs/2203.15556">Hoffmann et al., Training Compute-Optimal Large Language Models</link>; <link href="https://arxiv.org/abs/2109.07740">Ghorbani et al., Scaling Laws for Neural Machine Translation</link>. Reproduction: scaling_estimate.py and experiments/scaling-analysis/.','SmallText'),
        PageBreak(),p('Final-checkpoint T4 inference','Title'),
        p('Protocol','Head'),
        p('Prepare and verify model weights, tokenizer, official 1,919 questions and source on CPU. Save a hash-verified bundle to Drive, export and verify its ready record, then allocate one T4. Compare FP32 batch sizes 64, 128 and 256, twice with reversed order. Keep greedy generation, cached decoder state, length sorting, syntax-only beam retry and input/output limits 128/192 unchanged. Save every exact output, token provenance, hash and measurement to Drive. The two-model run omits optional profiling to keep GPU use bounded.'),
        p('Generation time includes tokenization, sorting, generation, retries, decoding and local CSV writing. Full-call time additionally covers test reading, model hashing, provenance writing and verification. Model load and a small warm-up are measured separately. Drive archival is outside the generation timer. The experiment has a 600-second soft budget and a bounded child-process timeout; the notebook requests GPU release in finally.'),
    ]
    if t4 and t4.get('completed'):
        rows = [['Model','Batch','Median, s','Slowest, s','Same outputs as 64']]
        for label, data in [('9.97M, 180k',t4),('42.48M, 6k',large_t4)]:
            if data and data.get('completed'):
                for item in data['summary']:
                    rows.append([label,item['batch_size'],f"{item['median_seconds']:.3f}",
                                 f"{item['max_seconds']:.3f}",item['all_outputs_equal_baseline']])
        story += [table(rows,[35*mm,20*mm,35*mm,35*mm,45*mm]),
                  p(f"Device: {escape(t4['gpu'])}; PyTorch {escape(t4['torch'])}; Transformers {escape(t4['transformers'])}. Model loading took {t4['model_load_seconds']:.3f} seconds and warm-up took {t4['warmup_seconds']:.3f} seconds. Changed programs require separate quality evaluation; byte-identical outputs preserve the same predictions."),
                  p('Batch 256 is selected for these immutable weights. Both models finish far below 180 seconds. Every run matches its already-scored JSONL exactly: 53.85% for the established model and 18.96% for the 6k larger pilot. Duplicate bot submissions were avoided. Later weights need a fresh timing and output check.'),
                  p('Full runs, per-call times, memory peaks, provenance and output hashes are retained in experiments/t4-comparison-v1 and its Drive archive. Reproduce with hw2_t4_comparison_colab.ipynb; the main inference notebook now selects batch 256 for the established checkpoint. The completed T4 runtime was manually released after its asynchronous release request left it connected.')]
    else:
        story.append(p('The first T4 attempt stopped before generation because drive.mount failed. Its notebook failure handler released the GPU. A refreshed CPU-only session reproduced the failure. On 16 October the user restored Drive access on CPU; a subsequent four-model GPU pilot saved and exported all checkpoints successfully. Actual T4 inference remains unmeasured for the current candidates, so no T4 speedup or compliance is claimed.'))
    pilot_root = ROOT / 'experiments/short-pilots-v1'
    if (pilot_root / 'results.json').exists():
        pilots = json.loads((pilot_root / 'results.json').read_text())
        rows = [['Parameters / layers', 'Peak LR', 'Val. loss', 'Held-out / 245', 'Public / 960', 'Train + export, s']]
        for x in pilots:
            receipt = json.loads((ROOT / 'submissions' / f"2026-10-16-pilot-{x['name']}-step-002000.json").read_text())
            layers = '6/4' if x['name'].startswith('encoder6') else '4/4'
            rows.append([f"{x['parameters']/1e6:.2f}M, {layers}", '.0003' if x['name'].endswith('3e4') else '.001',
                         f"{x['validation']['token_loss']:.4f}", x['numeric']['numeric_correct'],
                         f"{receipt['public_correct']} ({receipt['public_percent']:.2f}%)", f"{x['train_and_export_seconds']:.2f}"])
        story += [PageBreak(), p('Short architecture and rate pilots', 'Title'),
            p('Four fresh models, 2,000 updates each', 'Head'),
            p('The screen kept seed 42, batch 128, BF16 training, the same mixed data stream, tokenizer and 128/192 token limits. All runs used 200 warm-up updates and a short cosine schedule. Width, encoder depth and learning rate were varied in separate comparisons. One immutable checkpoint per trial was saved in Drive, exported with hashes and selected token IDs, verified locally, then scored once by the bot. Source: 1f1558515ce00d1be29f032bac843bcbfc5b6bf4.'),
            table(rows, [38*mm,19*mm,23*mm,25*mm,33*mm,32*mm]),
            p('Keep the 9.97M model at .001 and the 42.48M model at .0003. The small model has lower token loss and lower cost; the larger finalist has higher held-out execution agreement, confirmed by the bot. Lowering the rate within the 42M architecture improved all quality measures. The larger high-rate models do not justify continuation from this screen.'),
            p('These are early-learning results from one seed, not final convergence estimates. Even the best pilot remains below the established 180k model at 53.85%. The bot reported public rank 10/15 on 16 October; earlier rank 9/15 was an earlier snapshot. No private score is available. The later T4 comparison covers the established model and the larger 6k finalist.'),
            p('Time and next comparison', 'Head'),
            p('The driver took 718.51 seconds after data restoration; its launcher record spans 758.79 seconds including restoration and imports. Across 716 GPU samples, mean activity was 72.90%, median 97%, and peak memory 18,627 MiB. Activity is not achieved compute throughput; startup, I/O and execution evaluation contribute low samples. Batch-size speed still needs a matched benchmark.'),
            p('Continue both finalists from 2k to 6k with their optimizer moments, batch 128 and respective learning rates. Both receive the same new deterministic sample stream, a restarted 4k cosine schedule and 200 warm-up updates. Save and score the 6k checkpoint for each. Check actual T4 inference before long training. The four-way GPU was released manually after the automatic notebook request left it connected; file work resumed on CPU.'),
            p('Evidence: experiments/short-pilots-v1, its selection.md and export.json, four submission receipts, finalist_pilots.py and hw2_finalist_pilots_colab.ipynb. Full weights, optimizer states and exact predictions remain in Google Drive.', 'SmallText')]
    finalist_root = ROOT / 'experiments/finalist-pilots-v1'
    if (finalist_root / 'results.json').exists():
        finalists = json.loads((finalist_root / 'results.json').read_text())
        rows = [['Model', 'Val. loss', 'Held-out /245', 'Public /960', 'Added train + export, s']]
        for x in finalists:
            receipt = json.loads((ROOT / 'submissions' / f"2026-10-16-finalist-{x['name']}-step-006000.json").read_text())
            rows.append(['9.97M' if x['name'].startswith('small') else '42.48M',
                         f"{x['validation']['token_loss']:.4f}", x['numeric']['numeric_correct'],
                         f"{receipt['public_correct']} ({receipt['public_percent']:.2f}%)",
                         f"{x['train_and_export_seconds']:.2f}"])
        story += [PageBreak(), p('Finalists after 6,000 updates', 'Title'),
            p('Matched continuation', 'Head'),
            p('Both models resumed their own 2k weights and Adam moments. Batch 128, BF16, original targets, the same new mixed sample stream, a restarted 4k cosine schedule and 200 warm-up updates were held fixed. Peak rates remained .001 for 9.97M and .0003 for 42.48M. Source: 10db3cda45ba777e04e4195f003a6b1db5254b24.'),
            table(rows, [25*mm,30*mm,30*mm,40*mm,45*mm]),
            p('The larger model leads execution accuracy in both evaluations; the smaller model has lower next-token loss and lower cost. The public difference is 28 answers (2.92 percentage points), while the held-out difference is only three answers. One seed and early checkpoints cannot establish an optimal model size or statistical significance. Both remain below the established 180k model at 53.85%.'),
            p('Measured cost', 'Head'),
            p('The added training loop took 160.2 seconds for 9.97M and 333.3 seconds for 42.48M. The wider timings in the table include startup, validation, checkpoint saves and all-test generation. At the larger model\'s measured rate, a corpus-sized amount of examples would take about 2.1 hours of training loop time at batch 128, before evaluation and archival. This improves on the older 3.8-4.4-hour parameter-linear estimate but remains a short-run extrapolation.'),
            p('Across 583 GPU samples, median activity was 96%, mean activity 72.15%, and peak used memory 18,631 MiB. Activity includes I/O and numeric evaluation and does not measure achieved compute efficiency. A separate batch probe uses the actual encoded/prefetched training input path, equal examples, reversed rounds and timing-boundary synchronization.'),
            p('Checkpoint and evaluation retention', 'Head'),
            p('Both immutable 6k checkpoints, optimizer states, predictions and validation records were saved to Drive. The evidence archive was created there before download. Local checks verified hashes, all 1,919 official question positions and exact decoding of selected token IDs. Each checkpoint was submitted once to the course bot. Receipts are retained separately from the pre-submission archive. No answer text was edited.'),
            p('The G4 runtime was manually released after its asynchronous release request left it connected. Standard CPU was used to stage the established 180k model and the larger 6k finalist for actual T4 inference. A faster training batch must also pass a short equal-example quality check before being adopted for full training.'),
            p('Evidence: experiments/finalist-pilots-v1, its selection.md and export.json, the two 6k submission receipts, experiments/t4-comparison-v1/ready.json and benchmark_production_batches.py.', 'SmallText')]
    grouped = ROOT / 'experiments/grouped-42m-20k-v1'
    if (grouped/'results.json').exists():
        selection = json.loads((grouped/'selection.json').read_text())
        speed = json.loads((grouped/'batch-probe/results.json').read_text())
        stages = json.loads((grouped/'results.json').read_text())
        execution = json.loads((grouped/'execution.json').read_text())
        speed_rows = [['Batch', 'Median examples/s', 'Peak allocated GiB']]
        for size, rate in speed['median_examples_per_second'].items():
            peak = max(r['peak_allocated_gib'] for r in speed['runs'] if r['batch_size']==int(size))
            speed_rows.append([size, f'{rate:.1f}', f'{peak:.2f}'])
        quality_rows = [['Batch', 'Added examples', 'Token loss', 'Held-out /245', 'Public /960']]
        for stage in selection['quality_results']:
            checkpoint = Path(stage['checkpoint']).name
            receipt_path = grouped/stage['name']/checkpoint/'bot_receipt.json'
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            quality_rows.append([stage['batch_size'], f"{stage['added_examples']:,}",
                f"{stage['validation']['token_loss']:.4f}", stage['numeric']['numeric_correct'],
                receipt.get('public_correct', 'pending')])
        checkpoint_rows = [['Continuation step', 'Token loss', 'Public correct /960']]
        for marker in sorted((grouped/'continuation').glob('step-*/results.complete.json')):
            checkpoint = marker.parent
            result = json.loads(marker.read_text())
            validation = json.loads((checkpoint/'validation.json').read_text())
            receipt_path = checkpoint/'bot_receipt.json'
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            checkpoint_rows.append([f"{result['step']:,}", f"{validation['token_loss']:.4f}",
                                    receipt.get('public_correct','pending')])
        story += [PageBreak(),p('Larger model: speed and quality','Title'),
            p('The 42.48M model used one prepared G4 session for a speed probe, short quality checks and a capped continuation. All eight probe passes used 32,768 original examples and 2,419,826 target tokens; the second round reversed the batch order. Probe weights were discarded.'),
            table(speed_rows,[35*mm,70*mm,65*mm]),
            p('Batch 64 was 10.55% faster than 128. Both quality pilots resumed the same own 6k weights and optimizer state, with fixed added example counts, BF16 and peak rate .0003. The predefined gate required no fewer correct held-out executions and token loss within 1% of the control.'),
            table(quality_rows,[20*mm,40*mm,35*mm,35*mm,40*mm]),
            p(f"Batch {selection['batch_size']} was retained. Lower token loss at batch 64 did not compensate for its lower execution accuracy. The chosen lineage continued to 2,560,000 total example presentations, equivalent to 20,000 updates at batch 128; this is not a full pass over the 11.84M-row corpus."),
            table(checkpoint_rows,[55*mm,55*mm,60*mm]),
            p(f"The 10k public regression followed a learning-rate restart; causality was not established. The final held-out result was {stages[-1]['numeric']['numeric_correct']}/245. The grouped driver took {execution['elapsed_seconds']/60:.2f} minutes including probes, preparation, evaluation and Drive saves. Each checkpoint retained exact token provenance and a separate bot receipt; no program was rewritten."),
            p('Decoding opportunity','Head'),
            p('On the established 180k model, four-beam decoding improved held-out numeric agreement from 131/245 to 137/245 and reduced execution failures from 19 to 11. CPU generation took 10.61 versus 50.37 seconds. The subsequent full test export scored 544/960 (56.67%), versus 517/960 (53.85%) for greedy decoding. All 1,919 questions took 503.50 seconds on CPU at batch 64, FP32. T4 timing for this output remains unverified and is deferred. Every program matches its saved model token decode; no program was rewritten.'),
            p('Source: 85532a78132c37ceb9cbf3f6cfff5c478d69258f. Evidence: experiments/grouped-42m-20k-v1, experiments/decoding-180k and experiments/beam4-quality-180k-v1 (source d1aec6e). Results were downloaded with the authenticated local Drive CLI, avoiding Colab allocation for export.','SmallText')]
    full = ROOT/'experiments/full-42m-100k-v1'
    if (full/'started.json').exists():
        full_rows = [['Step', 'Token loss', 'Public correct /960', 'Public %']]
        full_rows.append(['20,000 (parent)', '0.6167', '320', '33.33'])
        for receipt_path in sorted(full.glob('step-*/bot_receipt.json')):
            receipt = json.loads(receipt_path.read_text())
            val = json.loads((receipt_path.parent/'validation.json').read_text())
            full_rows.append([f"{receipt['step']:,}", f"{val['token_loss']:.4f}",
                              receipt['public_correct'], f"{receipt['public_percent']:.2f}"])
        story += [PageBreak(), p('Full model and decoding checks','Title'),
            p('The 42.48M model resumes its own 20k weights and Adam moments. The original target was 100k updates; the user requested a shift away from scaling, so the run was intentionally stopped after saving 70k. The GPU session was released. Settings were batch 128, BF16, peak learning rate .00015, 500 warmup updates, cosine decay and the encoded/prefetched input pipeline. Halving the prior peak is a restart hypothesis, not a fitted optimum. The first 30k and 40k bot scores are review points.'),
            p('Before allocating G4, CPU checks verified all dataset hashes, loaded the model and optimizer, checked finite optimizer moments and saved a verified Drive bundle. At startup the GPU showed 98% activity and 18,951 MiB used out of 97,887 MiB. This is an observed snapshot, not a utilization guarantee. Each 10k checkpoint saves weights, optimizer state, exact predictions and a separate bot receipt.'),
            table(full_rows, [43*mm,40*mm,47*mm,40*mm]),
            p('At 3.84M example presentations, the larger model scored 409/960, versus 273/960 for the smaller model at step 40k. Different learning-rate schedules and data streams prevent a causal size-only comparison. These are scored checkpoints, not a completed target run. These historical exports use greedy decoding with syntax-only beam retries. No test labels are used for training. Source: 69829cbe; evidence: experiments/full-42m-100k-v1.','SmallText')]
        vote = ROOT/'experiments/candidate-vote-180k-v1'
        if (vote/'selection.json').exists():
            pilot = json.loads((vote/'pilot/results.json').read_text())
            confirm = json.loads((vote/'confirmation/results.json').read_text())
            vote_rows = [['Selection', 'Rows 512:768 (247 usable)', 'Rows 768:1024 (246 usable)']]
            for key, label in [('top_beam','Top beam'), ('first_executable','First executable'),
                               ('strict_majority','Strict majority')]:
                vote_rows.append([label, pilot['metrics'][key]['numeric_correct'],
                                  confirm['metrics'][key]['numeric_correct']])
            story += [p('Execution agreement without new training','Head'),
                p('Using the fixed 9.97M, 180k model, generate four deterministic beam candidates and execute their solve() functions in the restricted diagnostic runner. Select an unchanged program from a strict majority of all four numeric answers; otherwise retain the first executable candidate, breaking ties by model rank. References enter only after selection, for evaluation. No program is rewritten or added to training.'),
                table(vote_rows,[50*mm,60*mm,60*mm]),
                p('Strict majority passed the predefined positive-gain gate on both subsets: three extra correct answers across 493 usable references. This small exploratory gain is not statistical significance. First-executable selection failed the confirmation gate. Candidate and selected token decodes were checked after Drive export. The historical validation-copy caveat still applies to the 180k lineage.'),
                p('Both tests ran on four CPU cores of the 48-core training host with CUDA hidden; GPU training continued. The full-test export scored 548/960 (57.08%), four more than beam-only decoding; CPU generation and selection took 279.43 seconds. T4 timing is unverified. Inspiration: <link href="https://arxiv.org/abs/2203.11171">Wang et al., Self-Consistency</link>; our deterministic beam experiment differs from their sampling method. Sources: f047d7d (pilot), 124e46d (test export). Evidence: experiments/candidate-vote-180k-v1 and candidate-vote-test-180k-v1.','SmallText')]
    greedy_paths = sorted((ROOT/'experiments').glob('strict-greedy-42m*/step-*/bot_receipt.json'))
    greedy_paths += sorted((ROOT/'experiments/checkpoint-average-70k-v1').glob('exports/baseline/step-*/bot_receipt.json'))
    if greedy_paths:
        greedy_rows = [['Step', 'Token loss', 'Public correct /960', 'Public %']]
        for receipt_path in sorted(greedy_paths, key=lambda path:json.loads(path.read_text())['step']):
            receipt = json.loads(receipt_path.read_text())
            val = json.loads((receipt_path.parent/'validation.json').read_text())
            greedy_rows.append([f"{receipt['step']:,}", f"{val['token_loss']:.4f}", receipt['public_correct'], f"{receipt['public_percent']:.2f}"])
        story += [PageBreak(), p('Selected policy: strict greedy decoding','Title'),
            p('The user selected greedy decoding on 16 October. All new evaluated exports use num_beams=1 and do_sample=False, with syntax retries and execution voting disabled. Historical results are preserved with their original settings. Training stopped after its durable 70k checkpoint; separate CPU export processes apply strict greedy to saved weights.'),
            table(greedy_rows, [43*mm,40*mm,47*mm,40*mm]),
            p('The 50k strict greedy output scored 519/960 (54.06%), matching the public score of its earlier retry-enabled output. Two of the 1,919 output strings differed; the comparison also changed device and batch size, so this is not a controlled attribution of the difference to retries alone. CPU generation took 99.67 seconds at batch 64. This is not T4 timing.'),
            p('The default-decoder regression test supplies malformed Python and verifies that it is preserved with no extra generation calls. Full exports verify every program against saved token IDs, the official question-order hash, checkpoint hash and source commit. Each output is first saved to Drive, exported with the authenticated CLI, checked again and submitted once. Receipts are read back from Drive byte-for-byte.'),
            p('Source: 8dcb74d2164a30c7b3169dcf7adf76114139761f. Evidence: experiments/decoding-policy.json, experiments/strict-greedy-42m-v1 and subsequent strict-greedy-42m checkpoint directories. The main inference notebook follows this policy; the earlier voting notebook is retained only for historical reproduction. First place remains unmet.','SmallText')]
    average = ROOT/'experiments/checkpoint-average-70k-v1'
    if (average/'selection.json').exists():
        selection = json.loads((average/'selection.json').read_text())
        rows = [['Model', 'Pilot correct /247', 'Execution failures']]
        for name, result in selection['pilot'].items():
            rows.append([name, result['numeric_correct'], result['prediction_failures']])
        story += [PageBreak(), p('Checkpoint averaging test','Title'),
            p('Arithmetic averaging combines our own 60k/70k or 50k/60k/70k weights into one model of unchanged size. It adds zero training updates and still generates one greedy candidate. All runs use CPU, FP32, batch 64 and two threads. The original Transformer paper also used checkpoint averaging, although its decoding settings differed.'),
            table(rows,[70*mm,50*mm,50*mm])]
        if 'confirmation' in selection:
            rows = [['Model', 'Confirmation correct /246', 'Execution failures']]
            for name, result in selection['confirmation'].items():
                rows.append([name, result['numeric_correct'], result['prediction_failures']])
            story += [table(rows,[70*mm,50*mm,50*mm])]
        story += [p(f"Predefined gate: strictly more numeric agreements and no more execution failures than 70k on both held-out samples. Selected candidate: {escape(str(selection['selected']))}; confirmed: {selection['confirmed']}. These are small exploratory samples, not a significance test or public pass@1. Public test answers never choose the average."),
            p('Both candidates and their parent hashes are retained on Drive. Original held-out training references are used only for evaluation, with no generated training targets or handwritten repairs. Source: 861ae9795086a67af530f8eb2fc873423f466fbd. Evidence: experiments/checkpoint-average-70k-v1. Reference: <link href="https://arxiv.org/html/1706.03762v7#S6.SS1">Vaswani et al., Attention Is All You Need, section 6.1</link>.','SmallText')]
        for receipt_path in sorted(average.glob('exports/*/step-*/bot_receipt.json')):
            receipt = json.loads(receipt_path.read_text())
            story += [p(f"Bot result for {escape(receipt['variant'])}: {receipt['public_correct']}/960 ({receipt['public_percent']:.2f}%). Exact file: {escape(receipt['filename'])}. T4 timing remains unverified.")]
    loss_pilot = ROOT/'experiments/numeric-loss-2k-v1'
    if (loss_pilot/'selection.json').exists():
        result = json.loads((loss_pilot/'selection.json').read_text())
        rows = [['Arm', 'Pilot agreements', 'Confirmation agreements', 'Public correct /960']]
        for name, stages in result['results'].items():
            receipt = loss_pilot/name/'step-072000/bot_receipt.json'
            public = json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
            rows.append([name] + [f"{stages[stage]['numeric_correct']}/{stages[stage]['usable_references']}" for stage in ['pilot','confirmation']] + [public])
        story += [PageBreak(), p('Short matched loss-weight test','Title'),
            p('Two independent 2,000-update continuations start from the same own 70k weights and optimizer state. Both use batch 128, seed 42, identical sampled data, BF16, peak learning rate .00003, 100 warmup steps and the same cosine schedule. Control uses the original loss. Numeric2 gives twice the weight to 305 of 8,192 tokenizer pieces matching numbers or arithmetic operators. Original programs are unchanged; matching pieces in comments and strings also receive weight. This is a heuristic, not a syntax-aware objective.'),
            table(rows,[30*mm,45*mm,55*mm,40*mm]),
            p(f"The predefined held-out gate requires more numeric agreements and no extra execution failures on both subsets. Confirmed: {result['confirmed']}. This is a short exploratory comparison on previously used validation questions; it is not statistical significance. Training losses across arms are not directly comparable. Uniform validation loss and executed-answer diagnostics remain comparable."),
            p('Five focused tests checked normalized weighting, ignored padding, piece selection and continuation scheduling. CPU checks verified all dataset files, model weights and optimizer moments before G4 allocation. Both checkpoints and exact greedy outputs were saved to Drive before local export, token verification and bot submission. The control scored 562/960 (58.54%) and numeric2 scored 573/960 (59.69%), an 11-answer public gain at equal training budget. Numeric2 was retained at that stage for reproducing the highest public score; it did not pass the held-out gate for extending training. Total experiment time was 453 seconds, median GPU activity 97%. Colab showed no active sessions after completion. No longer training run started. Source: 8755a8f4867683d940e9b34fa48365e3cf4f8c21; evidence: experiments/numeric-loss-2k-v1. T4 timing remains deferred.','SmallText')]
    screen = ROOT/'experiments/broad-screen-1k-v1'
    if (screen/'selection.json').exists():
        selection = json.loads((screen/'selection.json').read_text())
        rows = [['Arm', 'Held-out /738', 'Failures', 'Public /960']]
        for name, stages in selection['results'].items():
            receipt = screen/name/'step-071000/bot_receipt.json'
            public = json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
            rows.append([name, sum(v['numeric_correct'] for v in stages.values()),
                         sum(v['prediction_failures'] for v in stages.values()), public])
        story += [PageBreak(), p('Broad short experiment screen','Title'),
            p('Eight matched continuations start from the same 42.48M model at step 70,000, including its optimizer state. Each gets only 1,000 updates at batch 128. Seven candidates change one factor relative to control: numeric token weighting, label smoothing, equal question weight, dropout 0 or .05, doubled peak learning rate, or excluding targets longer than 192 tokens. Targets are never rewritten. Filtering refills complete batches but necessarily changes which examples are seen.'),
            table(rows,[55*mm,40*mm,30*mm,45*mm]),
            p('All outputs use strict greedy decoding. The shortlist requires positive total held-out gain, no negative gain on either validation subset, and no increase in total execution failures. At most three candidates proceed to a second-seed test against control. Combinations need their own short comparison before a long run; individual gains are not assumed to add together.'),
            p('Shortlist: '+escape(', '.join(selection['shortlist']) or 'none')+'. These previously reused validation questions provide exploratory evidence, not an untouched estimate of generalization. Public bot scores are recorded separately; individual public failure examples are not used to design targets or repairs.'),
            p('Seventeen focused checks passed before CPU verification and GPU allocation. Checkpoints, exact greedy predictions, source identity and GPU telemetry are saved to Drive before export. Source: 5de771fc79b3b1e84926aedd8292560e56d9fecd. Evidence: experiments/broad-screen-1k-v1. Label smoothing and dropout motivation: <link href="https://arxiv.org/html/1706.03762v7#S5.SS4">Vaswani et al., section 5.4</link>. Their usefulness here is determined by the measured trials.','SmallText')]
        if (screen/'execution.json').exists():
            execution=json.loads((screen/'execution.json').read_text())
            story += [p(f"Driver duration: {execution['elapsed_seconds']/60:.2f} minutes, including data restore, training, evaluation and saves. Colab allocation and Drive sign-in occur before this timer. No long run is launched automatically.")]
    audit = ROOT/'experiments/train-diversity-audit-v1'
    if (audit/'results.json').exists():
        r=json.loads((audit/'results.json').read_text())
        story += [PageBreak(),p('Training distribution audit','Title'),
            p('Before another GPU experiment, scan all 60 original training shards on CPU and exclude every exact copy of the 1,024 held-out question texts. Count exact questions using BLAKE2b-128 hashes and draw a fixed sample of 2,048 original pairs per shard for tokenizer statistics. No training targets are created or rewritten.'),
            table([['Measurement','Observed'],
                ['Eligible original rows',f"{r['total_retained_rows']:,}"],
                ['Distinct question hashes',f"{r['unique_question_hashes']:,}"],
                ['Excess duplicate rows',f"{100*r['excess_duplicate_row_fraction']:.2f}%"],
                ['Rows belonging to repeated questions',f"{100*r['rows_from_repeated_questions_fraction']:.2f}%"],
                ['Sample size',f"{r['sample_rows']:,}"],
                ['Questions exceeding 128 tokens',f"{100*r['source_over_128']/r['sample_rows']:.3f}%"],
                ['Programs exceeding 192 tokens',f"{100*r['target_over_192']/r['sample_rows']:.3f}%"],
                ['Median / mean program tokens',f"{r['target_tokens']['median']:.0f} / {r['target_tokens']['mean']:.2f}"]],[105*mm,65*mm]),
            p('Exact duplication and truncation affect a minority of rows. This does not prove filtering cannot help, but it does not establish a large data-cleaning opportunity. The audit is descriptive: it does not assess semantic correctness, normalized paraphrases, or the effect of changing the training distribution.'),
            p('A separate code inspection found that grouped batches always run in ascending input-length order within each window. An optional deterministic shuffle now changes only batch order, preserving every batch tensor and its padding cost. Thirteen focused checks passed, including unchanged default behavior and exact shuffled batch membership. This option is prepared for a short two-seed comparison; no score gain or completed training run is claimed.'),
            p(f"Audit computation: {r['elapsed_seconds']:.2f} seconds; CPU process, input restore and setup: 85.97 seconds. No GPU was allocated. Colab confirmed no active sessions after automatic release. The contemporaneous public leaderboard remains 8/15: 59.69% versus 78.33% for first place. Evidence: experiments/train-diversity-audit-v1; audit source: 6add585. The first-place objective remains unmet.",'SmallText')]
    ordering = ROOT/'experiments/batch-order-1k-v1'
    if (ordering/'selection.json').exists():
        r=json.loads((ordering/'selection.json').read_text())
        rows=[['Seed / order','Pilot /245','Confirm /493','Failures','Public /960']]
        for name, stages in r['results'].items():
            f=ordering/name/'step-071024/bot_receipt.json'
            public=json.loads(f.read_text())['public_correct'] if f.exists() else 'pending'
            rows.append([name,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],
                         sum(s['prediction_failures'] for s in stages.values()),public])
        story += [PageBreak(),p('Grouped batch order: two-seed test','Title'),
            p('Four continuations start from identical own 70k weights and Adam moments. Within each seed, the control processes length-grouped batches in ascending length order; the treatment shuffles these same completed batches within each 4,096-example window. Both retain identical batch membership, padding cost and original targets. Seeds 42 and 1042 change the data/dropout stream. Each arm gets 1,024 updates, batch 128, or 131,072 example presentations; the budget ends on a complete grouping window.'),
            table(rows,[49*mm,29*mm,35*mm,25*mm,32*mm]),
            p('The predefined gate requires positive total held-out gain in each seed, no negative gain in either subset for either seed, and no increase in execution failures pooled across seeds. Confirmed: '+str(r['confirmed'])+'. Reused questions across seeds are not independent test observations, and all validation remains exploratory.'),
            p('The shuffled order gained 3 held-out correct answers in seed 42 but lost 1 in seed 1042, with 3 additional pooled execution failures. Public accuracy fell by 11 correct answers in each pair. No long extension is justified. The retained best remains 573/960 (59.69%), rank 8/15.'),
            p('All arms use unchanged architecture, original token loss, peak learning rate .00003, 50 warmup updates and strict greedy decoding. Fifteen focused tests passed, including exact preservation of shuffled batch tensors, unchanged default behavior, checkpoint boundaries and paired selection. Every checkpoint and test output is saved to Drive before export and verification. Source: 862fa6f491c6c1bd4c7977d688aa37deaf2c54db; evidence: experiments/batch-order-1k-v1.','SmallText')]
        if (ordering/'execution.json').exists():
            e=json.loads((ordering/'execution.json').read_text())
            story += [p(f"Driver duration: {e['elapsed_seconds']/60:.2f} minutes, excluding allocation and authorization before the driver. Median sampled GPU utilization was 97%; mean was 63.79%, including restoration, CPU scoring and saving. Colab confirmed no active sessions after automatic release. No long extension was started.")]
    digits = ROOT/'experiments/digit-screen-1k-v1'
    if (digits/'selection.json').exists():
        r=json.loads((digits/'selection.json').read_text())
        audit=json.loads((digits/'tokenization-audit.json').read_text())
        rows=[['Tokenizer','Pilot /245','Confirm /493','Failures','Public /960']]
        for name, stages in r['results'].items():
            f=digits/name/'step-071024/bot_receipt.json'
            public=json.loads(f.read_text())['public_correct'] if f.exists() else 'pending'
            rows.append([name,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],
                         sum(s['prediction_failures'] for s in stages.values()),public])
        story += [PageBreak(),p('Individual digits: short adaptation test','Title'),
            p('The treatment adds individual digit boundaries before BPE merges while retaining the original learned vocabulary, token IDs, model architecture and target text. The control and treatment continue from identical own 70k weights and Adam moments for 1,024 updates each, with batch 128, peak learning rate .00003 and 50 warmup updates. Both use strict greedy generation. The same original example pool is consumed at the complete-window boundary; longer sequences change grouping and padding.'),
            p(f"CPU audit: {audit['rows']:,} original pairs sampled across all 60 shards, excluding held-out question copies. Both tokenizers recover every original text exactly; save/reload and unchanged vocabulary IDs passed. Question mean length rises from 46.97 to 51.85 tokens; program mean from 74.52 to 80.94. At the unchanged 128/192 limits, question truncations rise from 218 to 446 and program truncations from 2,375 to 3,005."),
            table(rows,[49*mm,29*mm,35*mm,25*mm,32*mm]),
            p('The predefined gate requires improvement on both held-out subsets and no increase in total execution failures. Shortlisted for a second-seed test: '+(', '.join(r['shortlist']) or 'none')+'. No long extension is automatic. This tests adaptation of an existing BPE model, not a model trained from scratch with digit boundaries. Repeated validation is exploratory.'),
            p('The treatment lost 55 held-out agreements and added 6 execution failures. Public accuracy fell by 50/960 (5.21 percentage points). Reject this adaptation; the best remains 573/960 (59.69%), rank 8/15. The original control reproduces previous correct counts but 11 of 1,919 program strings differ, so bitwise repeatability is not established.'),
            p('Motivation: Singh and Strouse, Tokenization counts (2024), https://arxiv.org/abs/2402.14903. That paper concerns arithmetic tokenization and does not establish a benefit for this Python-generation task. No paper examples or additional targets enter training.','SmallText'),
            p('Eighteen focused checks passed before launch. Every checkpoint and its original test output was saved to Drive before export and exact verification. Training source: f9676760c37607102647c7d0b792810f81f1e8f1. Evidence: experiments/digit-screen-1k-v1.','SmallText')]
        if (digits/'execution.json').exists():
            e=json.loads((digits/'execution.json').read_text())
            story += [p(f"GPU driver duration: {e['elapsed_seconds']/60:.2f} minutes, excluding allocation and authorization. CPU preparation took 149.47 seconds, including the 65.65-second tokenizer audit. Colab confirmed no active sessions after automatic release; T4 timing remains unverified.")]
    focal = ROOT/'experiments/focal-screen-1k-v1'
    if (focal/'selection.json').exists():
        r=json.loads((focal/'selection.json').read_text())
        rows=[['Loss','Pilot /245','Confirm /493','Failures','Public /960']]
        for name, stages in r['results'].items():
            f=focal/name/'step-071024/bot_receipt.json'
            public=json.loads(f.read_text())['public_correct'] if f.exists() else 'pending'
            rows.append([name,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(s['prediction_failures'] for s in stages.values()),public])
        story += [PageBreak(),p('Focal token loss: short matched screen','Title'),
            p('Three continuations start from the same own 70k model and Adam state. Each receives 1,024 updates, batch 128, peak learning rate .00003, 50 warmup updates and unchanged original data, vocabulary and programs. The control uses cross-entropy. Focal1 and focal2 multiply each original target-token loss by (1-p) raised to gamma=1 or 2, where p is its predicted probability. The gradient includes this factor; alpha=1. The denominator remains the original valid-token weight. Greedy decoding is unchanged.'),
            table(rows,[49*mm,29*mm,35*mm,25*mm,32*mm]),
            p('The predefined gate requires improvement on both held-out subsets without more total execution failures. Shortlisted for second-seed confirmation: '+(', '.join(r['shortlist']) or 'none')+'. No automatic combination or long run. Focal modulation changes gradient magnitude as well as emphasis, so identical nominal learning rates do not imply identical effective steps. Reused validation remains exploratory.'),
            p('Evaluation now saves the exact generated programs and original references before scoring. Separate CPU-only scorers run while later GPU work proceeds. Payload hashes and row order are verified before scoring; the scoring branch imports no torch and does not load model weights. Process failures propagate to the driver, and pending processes are cleaned up. Twenty focused checks passed, including focal values/gradients, padding, score preservation and failure handling.'),
            p('The verified dataset and parent archive from the earlier CPU preflight are reused; new source and launch files are uploaded to Drive and read back before GPU allocation. Archive hashes are rechecked on restore. Checkpoints and original test outputs are saved to Drive before export and bot scoring. Source: 86c3bfb603eeb2f20c7e3b4cf7dd3f246b084d93; evidence: experiments/focal-screen-1k-v1.','SmallText'),
            p('Motivation: Lin et al., Focal Loss for Dense Object Detection, ICCV 2017. Its object-detection results do not establish a benefit for this sequence-generation task. No external examples or additional training targets are used.','SmallText')]
        if (focal/'execution.json').exists():
            e=json.loads((focal/'execution.json').read_text())
            story += [p('Focal1 gains four held-out agreements with one fewer execution failure, but its public score is two answers below control. Focal2 fails the held-out gate. No long run is justified; focal1 is only a candidate for short second-seed confirmation. The best remains 573/960 (59.69%), rank 8/15.'),
                p(f"Driver duration: {e['elapsed_seconds']/60:.2f} minutes, excluding allocation and Drive authorization. Median GPU activity was 97%, mean 65.68%. Only 3 of 34 samples during CPU scoring show nonzero GPU activity, none at 90%: overlap was mostly with setup, and speedup is not established. Colab confirmed no active sessions. All three bot receipts are saved and verified in Drive. T4 timing remains unverified.")]
    confirmation = ROOT/'experiments/focal-confirm-1k-v1'
    if (confirmation/'selection.json').exists():
        r=json.loads((confirmation/'selection.json').read_text())
        rows=[['Seed / loss','Pilot /245','Confirm /493','Failures','Public /960']]
        for seed, folder in [(42,focal),(1042,confirmation)]:
            values=json.loads((folder/'selection.json').read_text())['results']
            for name in ['control','focal1']:
                stages=values[name]
                receipt=folder/name/'step-071024/bot_receipt.json'
                public=json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
                rows.append([f'{seed} / {name}',stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(s['prediction_failures'] for s in stages.values()),public])
        story += [PageBreak(),p('Focal loss: second-seed confirmation','Title'),
            p('The gamma-one candidate alone is repeated against control with seed 1042. Both new arms start from the original own 70k parent and Adam state, not from the first-seed results. Each receives 1,024 updates with the same batch size, learning-rate schedule, data pipeline and strict greedy decoding. Seed 42 results are reused without retraining.'),
            table(rows,[49*mm,29*mm,35*mm,25*mm,32*mm]),
            p('The predefined confirmation gate requires positive gains on both held-out subsets in each seed, with no increase in total execution failures in either seed. Confirmed: '+str(r['confirmed'])+'. Public scores do not enter this gate. The same held-out questions are reused across seeds, so this is a robustness check rather than an independent validation set.'),
            p('Twenty-one focused checks passed, including rejection of zero subset gains, extra execution failures and a failing first seed. The source archive includes the exact first-seed selection evidence and records its hash. Unchanged data and parent archives reuse the earlier CPU preflight; source and launcher are uploaded and read back before GPU allocation. Each new checkpoint and its predictions are saved to Drive before export and bot evaluation.','SmallText'),
            p('Source: bbf58a9376348573d28a72b798113fa0d6bab6a9. Evidence: experiments/focal-confirm-1k-v1. No additional training examples, target rewrites or handwritten output repairs are used. No long run starts automatically. T4 timing remains deferred.','SmallText')]
        if (confirmation/'conclusion.json').exists():
            conclusion=json.loads((confirmation/'conclusion.json').read_text())
            story += [p(conclusion['decision']),p(conclusion['compute'])]
    cpu_audit = ROOT/'experiments/cpu-data-tokenizer-audit-v1'
    if (cpu_audit/'tokenizers/results.json').exists():
        r=json.loads((cpu_audit/'tokenizers/results.json').read_text())
        rows=[['Tokenizer','Question mean','Code mean','Code >192','Code delta']]
        for name,values in r['results'].items():
            rows.append([name,f"{values['question']['mean_tokens']:.2f}",f"{values['code']['mean_tokens']:.2f}",
                         values['code']['over_limit'],f"{values['code_token_change_vs_original_percent']:+.2f}%"])
        story += [PageBreak(),p('Python tokenizer: CPU comparison','Title'),
            p(f"At the user's suggestion, fit four new tokenizers on {r['fit_rows']:,} original question/program pairs sampled across all 60 shards. Evaluate on {r['evaluation_rows']:,} separate exact question texts; remove all held-out question copies from fitting and this audit. The existing 8,192-token byte BPE is retained as a reference. It was previously trained on shard 0, so its compression comparison is not an untouched generalization evaluation."),
            table(rows,[44*mm,35*mm,31*mm,30*mm,30*mm]),
            p('Balanced8192 sees each original question and code once. Code8192 repeats the unchanged original code three times per question during tokenizer fitting. Python8192 adds explicit lexical-style boundaries for identifiers, numeric strings, operators, whitespace and indentation. Code16384 doubles the vocabulary with the same code emphasis. No pretrained vocabulary, new examples or rewritten training targets are used.'),
            p('Every candidate is serialized, reloaded and checked for exact recovery of questions and code, including significant whitespace. Unknown tokens, number fragmentation and truncation counts are recorded. Shorter sequences do not prove better executed-answer accuracy. A different vocabulary cannot silently replace token meanings in the trained checkpoint; any deployment requires a matched training experiment.'),
            p('Five focused local checks passed, including saved tokenizer round trips with tabs, CRLF and Unicode, disjoint question sampling, unchanged original Parquet files and separation of numeric conflict from unsupported execution. Motivation: Hugging Face CodeParrot methodology, https://github.com/huggingface/blog/blob/main/codeparrot.md. That work motivates a code-specific tokenizer; it does not prove a gain on this task.','SmallText'),
            p('Source: 52cccd19da35f70d023cf63f2aeb0d161524de18. Evidence: experiments/cpu-data-tokenizer-audit-v1/tokenizers. All work uses CPU; artifacts are saved to Drive before export.','SmallText')]
        if (cpu_audit/'conclusion.json').exists():
            conclusion=json.loads((cpu_audit/'conclusion.json').read_text())
            story += [p(conclusion['tokenizer_decision'])]
    if (cpu_audit/'consistency/results.json').exists():
        r=json.loads((cpu_audit/'consistency/results.json').read_text())
        rows=[['Measurement','Count'],['Eligible original rows',f"{r['eligible_rows']:,}"],
              ['Sampled exact question groups',r['sampled_question_groups']],
              ['Repeated question groups',r['repeated_question_groups']],
              ['Repeated groups with identical program',r['identical_program_groups']],
              ['Repeated groups with different programs',r['different_program_groups']],
              ['Tested groups / programs',f"{r['tested_groups']} / {r['tested_programs']}"],
              ['Groups with program cap',r['capped_groups']]]
        rows += [[name,count] for name,count in r['categories'].items()]
        story += [PageBreak(),p('Original target consistency: CPU audit','Title'),
            p('Sample exact training-question groups using a fixed keyed hash, independently of public/test examples. Scan all original shards and exclude every held-out question copy. Among repeated questions with different original programs, test at most 512 groups and eight distinct programs per group. Execute only original programs with the existing restricted diagnostic runner; no values are converted into training targets.'),
            table(rows,[115*mm,55*mm]),
            p('Numeric conflict means at least two executable original programs disagree within the existing 1e-6 tolerance. It does not identify which program is correct. Agreement also does not prove correctness. Unsupported or failed execution remains unresolved rather than an automatic reason to discard a target. Groups are sampled by question, not weighted by frequency; the result cannot be read as the fraction of all training rows that are wrong.'),
            p('The audit saves question/program hashes, original shard/row positions, statuses and aggregate comparisons. It leaves original files unchanged and emits no training target file. Evidence: experiments/cpu-data-tokenizer-audit-v1/consistency.','SmallText')]
        if (cpu_audit/'conclusion.json').exists():
            conclusion=json.loads((cpu_audit/'conclusion.json').read_text())
            story += [p(conclusion['consistency_decision']),p(conclusion['compute'])]
    cleanup = ROOT/'experiments/code-cleanup-cpu-v1'
    if (cleanup/'results.json').exists():
        r=json.loads((cleanup/'results.json').read_text())
        story += [PageBreak(),p('Comments and empty lines: CPU preflight','Title'),
            p('At the user\'s suggestion, test removing non-executing text from existing training targets. The cleaner uses Python tokenization to distinguish comments from hash characters inside strings. Empty lines inside strings and explicit line continuations are preserved. Every accepted change must compile and have the same complete parsed Python structure, excluding source positions. Invalid or structurally changed programs fall back to the original.'),
            table([['Measurement','Result'],['Sampled training programs',r['rows']],['Programs with comments',r['counters']['rows_with_comments']],['Changed / rejected',f"{r['counters']['changed']} / {r['counters']['rejected']}"],['Mean target tokens: original / cleaned',f"{r['original_mean_tokens']:.2f} / {r['cleaned_mean_tokens']:.2f}"],['Target token change',f"{r['token_change_percent']:.2f}%"],['Targets exceeding 192: original / cleaned',f"{r['original_over_192']} / {r['cleaned_over_192']}"],['Executable pairs / identical numeric results',f"{r['execution_checks']['both_numeric']} / {r['execution_checks']['identical_numeric']}"],['Numeric / status mismatches',f"{r['execution_checks']['numeric_mismatch']} / {r['execution_checks']['status_mismatch']}"],['Unsupported execution pairs',r['execution_checks']['unsupported_pairs']]],[113*mm,57*mm]),
            p('The fixed sample comes from training shard 0 and excludes every exact copy of held-out question texts. This is a single-shard compression diagnostic, not evidence of improved model accuracy. Docstrings, pass statements, imports, assignments, variable names and calculations remain intact. Source or line-number introspection is outside the equivalence claim.'),
            p('The training experiment replaces each original target once; it creates no additional examples, alternate solutions or numerical-answer targets. Generated submissions remain exact greedy model output, with no cleanup after generation. Nine focused cleanup tests cover quoted hash characters, multiline strings, docstrings, Unicode, line continuation, invalid syntax, repeatability and preservation of the input example stream.','SmallText')]
    representation = ROOT/'experiments/representation-pilot-2k-v1'
    if (representation/'selection.json').exists():
        r=json.loads((representation/'selection.json').read_text())
        rows=[['Arm','Pilot /245','Confirm /493','Failures','Public /960']]
        for name,stages in r['results'].items():
            receipt=representation/name/'step-002048/bot_receipt.json'
            public=json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
            rows.append([name,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(s['prediction_failures'] for s in stages.values()),public])
        story += [PageBreak(),p('Representation: four short scratch trials','Title'),
            p('Compare original-tokenizer control, code-focused 8k vocabulary, code-focused 16k vocabulary, and comment/empty-line cleanup with the original tokenizer. Each receives 2,048 updates, batch 128, seed 42, learning rate 0.0003, and the same 262,134 original-example presentations (one batch has 118 rows). All models initialize at vocabulary 16,384; the 8k arms then shrink their tied embeddings. Their initial Transformer body hashes must match exactly. The dropout random stream is reset after shrinking.'),
            table(rows,[43*mm,30*mm,34*mm,28*mm,35*mm]),
            p('The 8k models have 42,481,152 parameters; the 16k model has 46,675,456. Source stream and grouping-window budgets match, but changed lengths can change sorting inside a window. Token-level losses across vocabularies or target representations are not directly comparable. Evaluation uses original held-out programs and strict greedy decoding.'),
            p('Predefined gate: improve both held-out subsets and add no total execution failures versus control. Candidates for a second-seed check: '+(', '.join(r['shortlist']) or 'none')+'. This early screen does not establish eventual performance after full training. Public results are recorded separately and do not enter the selection gate.'),
            p('Source: 558c3fd580c18d8f7a1ca4a68a8126138fd3e062. Thirty-two checks passed before allocation. All checkpoints and predictions are saved to Drive before local export and bot submission. A hard runtime bound and automatic release are installed. No long run is started automatically.','SmallText')]
        if (representation/'conclusion.json').exists():
            conclusion=json.loads((representation/'conclusion.json').read_text())
            story += [p(conclusion['decision']),p(conclusion['compute'])]
    token_confirm = ROOT/'experiments/tokenizer-confirm-2k-v1'
    if (token_confirm/'selection.json').exists():
        r=json.loads((token_confirm/'selection.json').read_text())
        rows=[['Seed / tokenizer','Pilot /245','Confirm /493','Failures','Public /960']]
        for seed,folder in [(42,representation),(1042,token_confirm)]:
            values=json.loads((folder/'selection.json').read_text())['results']
            for name in ['control','code8192']:
                stages=values[name];receipt=folder/name/'step-002048/bot_receipt.json'
                public=json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
                rows.append([f'{seed} / {name}',stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(v['prediction_failures'] for v in stages.values()),public])
        story += [PageBreak(),p('Code-focused 8k: second-seed check','Title'),
            p('Repeat the cheaper code-focused tokenizer against the original 8k control. Both new models start from random weights with seed 1042, receive 2,048 updates and use the same data, architecture, batch size, learning-rate schedule and strict greedy decoding. The tokenizer files are unchanged from the CPU fit and verified byte-for-byte. The initial Transformer bodies are identical within each seed; no target cleanup is applied.'),
            table(rows,[49*mm,29*mm,35*mm,25*mm,32*mm]),
            p('The confirmation gate requires strictly positive gains on both held-out subsets in each seed, with no increase in total execution failures in either seed. Confirmed: '+str(r['confirmed'])+'. Both seeds reuse the same held-out questions, so this checks seed robustness rather than independent validation or statistical significance. Public feedback is recorded separately and never converted into training targets.'),
            p('The first-seed evidence is copied into the source archive and hash-checked before the new run. Eighteen focused checks passed before allocation. Existing verified dataset, tokenizer and model-shape preflight evidence is reused without allocating a CPU Colab runtime. Source, launcher and manifest are saved to Drive and downloaded back before GPU allocation. Each completed checkpoint and exact model output is saved to Drive before bot evaluation.','SmallText'),
            p('Source: e463ac8cb58f6bb52608f6d3e067e3c1739020e5. Evidence: experiments/tokenizer-confirm-2k-v1. A 900-second driver bound and automatic runtime release are installed. A positive result permits considering a bounded extension, not an automatic full training run.','SmallText')]
        if (token_confirm/'conclusion.json').exists():
            conclusion=json.loads((token_confirm/'conclusion.json').read_text())
            story += [p(conclusion['decision']),p(conclusion['compute'])]
    frequency = ROOT/'experiments/frequency-index-cpu-v1'
    if (frequency/'index/results.json').exists():
        r=json.loads((frequency/'index/results.json').read_text())
        rows=[['Sampling power','Accepted rows','Unique questions','Expected repeat share']]
        for power,values in r['arms'].items():
            rows.append([power,f"{values['accepted_rows']:,}",f"{values['unique_questions']:,}",f"{100*values['expected_repeated_question_fraction']:.2f}%"])
        story += [PageBreak(),p('Original-data sampling: CPU preflight','Title'),
            p('Augmented question-program pairs are not used: the assignment prohibits additional training examples or targets. This experiment instead prepares inverse-frequency sampling of unchanged original rows. A question appearing f times is accepted with probability f to the power minus alpha. Alpha zero keeps the original sampler; 0.5 mildly reduces repetition; 1 applies full inverse-frequency weighting. All original program variants remain eligible. No answer is selected as correct and no program is rewritten.'),
            p(f"The CPU index covers {r['eligible_rows']:,} eligible rows and {r['unique_questions']:,} exact question strings across all 60 training shards. Every copy of a held-out question is excluded. One uint32 count is stored per original row; source and index checksums are verified before training. The existing mixed sampler draws equal candidates per shard, so this does not promise perfectly uniform question probabilities."),
            table(rows,[35*mm,40*mm,40*mm,55*mm]),
            p('The table measures one fixed candidate pool, not model accuracy or equal-budget training. Weighted pools are smaller; a future fixed-update trial draws further original rows to fill the same number of training batches. Candidate selection uses no public test information. Prior numeric disagreements do not identify the correct reference, so they are not used to delete or replace targets.'),
            p('Five focused tests cover exact cross-shard counts, held-out exclusion, checksum rejection, deterministic probabilities, original target preservation and unchanged zero-power behavior. Source: 1abd5e36874b45a5f9c28e8c4d06b3a0da617074. No GPU is allocated for this audit.','SmallText')]
        if (frequency/'stream-preflight.json').exists():
            values=json.loads((frequency/'stream-preflight.json').read_text())
            rows=[['Power','Original rows','Unique questions','CPU loading seconds']]
            for power,v in values.items():
                rows.append([power,f"{v['rows']:,}",f"{v['unique_questions']:,}",f"{v['seconds']:.2f}"])
            story += [table(rows,[25*mm,45*mm,45*mm,55*mm]),p('This timing covers data loading and sampling only, without tokenization or model training. It is not GPU throughput. Unique counts in a short stream need not increase: repeatedly occurring questions can fall outside the selected prefix by chance.')]
        if (frequency/'conclusion.json').exists():
            conclusion=json.loads((frequency/'conclusion.json').read_text())
            story += [p(conclusion['decision']),p(conclusion['compute'],'SmallText')]
    frequency_screen = ROOT/'experiments/frequency-screen-1k-v1'
    if (frequency_screen/'selection.json').exists():
        r=json.loads((frequency_screen/'selection.json').read_text())
        rows=[['Sampling','Pilot /245','Confirm /493','Failures','Public /960']]
        for name,stages in r['results'].items():
            receipt=frequency_screen/name/'step-073024/bot_receipt.json'
            public=json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
            rows.append([name,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(v['prediction_failures'] for v in stages.values()),public])
        story += [PageBreak(),p('Inverse-frequency sampling: short training test','Title'),
            p('Three matched 1,024-update continuations from the verified 72k numeric-weight-2 checkpoint. The control preserves original mixed sampling. Inverse_sqrt and inverse accept an original row with probability 1/sqrt(f) and 1/f, where f is its exact question frequency across the training split. No input or target text is changed, and no reference is selected as correct. All arms retain numeric-token weight 2, the same optimizer state, batch size 128, learning rate 3e-5, 50-step warmup and strict greedy decoding.'),
            table(rows,[42*mm,29*mm,35*mm,29*mm,35*mm]),
            p('A candidate must improve both held-out subsets and add no total execution failures relative to control. Shortlist: '+(', '.join(r['shortlist']) or 'none')+'. A survivor needs a second-seed check before a longer run. Public scores are recorded separately and do not enter this selection rule. Repeated use of these validation questions and selection of the parent by public score make these results exploratory.'),
            p('Eleven focused tests passed before allocation. CPU preflight verifies the parent weights, optimizer shapes and finite moments, resume step, a finite forward pass, the 60-shard frequency index and full batches at the actual continuation seed. Every model checkpoint and exact generated output is saved to Drive before export and bot evaluation. Training source: 0a3e2bfac3da70a5750efcf7233aa7506a6350e0. CPU data/model preflight used unchanged source 1d7ce0c3ec3ceae7d65e8d5153d308d5a5f69b8f. The first GPU attempt ended before training and left no checkpoint; Drive disconnected and its log was empty. Compute release was verified. The repaired launcher keeps its driver log locally and always requests compute release even if Drive export fails.','SmallText')]
        if (frequency_screen/'conclusion.json').exists():
            conclusion=json.loads((frequency_screen/'conclusion.json').read_text())
            story += [p(conclusion['decision']),p(conclusion['compute'])]
    comment_run = ROOT/'experiments/comment-continuation-1k-v1'
    if (comment_run/'selection.json').exists():
        r=json.loads((comment_run/'selection.json').read_text())
        rows=[['Targets','Pilot /245','Confirm /493','Failures','Public /960']]
        for name,stages in r['results'].items():
            receipt=comment_run/name/'step-073024/bot_receipt.json'
            public=json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
            rows.append([name,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(v['prediction_failures'] for v in stages.values()),public])
        story += [PageBreak(),p('Comment cleanup after learning: short matched trial','Title'),
            p('The earlier 2,048-step scratch experiment cannot establish whether removing comments helps an already trained model. This experiment compares two 1,024-update continuations from the same verified 72k numeric-weight-2 parent. Control retains original targets. Comments removes comments and empty lines outside strings only when parsing and compilation succeed and the Python syntax tree is identical. Each original target is replaced once; no examples or alternate solutions are added. Strings, docstrings, variable names, imports and executable statements are preserved.'),
            table(rows,[42*mm,29*mm,35*mm,29*mm,35*mm]),
            p('Both arms keep the original tokenizer, architecture, optimizer state, seed 42, batch 128, numeric-token weight 2, learning rate 3e-5, 50-step warmup and strict greedy decoding. Equal updates and complete batches preserve the example budget; target lengths change. Held-out references remain original. Shortlist: '+(', '.join(r['shortlist']) or 'none')+'. A candidate needs positive gains on both held-out subsets with no extra total execution failures, followed by another seed before any longer run. Public scores do not determine this gate.'),
            p('Thirteen focused tests passed. Verified parent/data CPU checks and the prior cleanup audit were reused; no additional CPU Colab runtime was allocated. Source, launcher and readiness records were saved to Drive and downloaded back before GPU allocation. Every checkpoint and exact test prediction was exported from Drive and verified before bot evaluation. The updated finalizer saves local diagnostic logs and requests release even if Drive fails.','SmallText')]
        if (comment_run/'conclusion.json').exists():
            conclusion=json.loads((comment_run/'conclusion.json').read_text())
            story += [p(conclusion['decision']),p(conclusion['compute'])]
    length_run = ROOT/'experiments/length-cap-cpu-v1'
    if (length_run/'conclusion.json').exists():
        conclusion=json.loads((length_run/'conclusion.json').read_text())
        story += [PageBreak(),p('Sequence limits: CPU-only inference test','Title'),
            p('A CPU audit of the best checkpoint tokenizer found only 3 of 1,919 test questions longer than the 128-token input limit (maximum 166). The first 1,024 held-out questions include 2 above this limit, and 5 reference programs exceed the 192-token target limit. In the exact scored best export, 21 generated sequences lack an end-of-sequence token. One row overlaps, giving 23 affected rows.'),
            p('Keep the same 42.48M model weights. Regenerate exactly those 23 rows on CPU with a 192-token input limit, a 384-token output limit, FP32, batch 8 and strict greedy decoding. Preserve the exact prior model-generated tokens on all other 1,896 rows. No code strings are edited, and no answer quality determines which result is retained. The full 1,919-row candidate is saved to Drive, exported and checked against generated token IDs before bot evaluation. This is a decoding test, with no training or GPU allocation.'),
            p(conclusion['decision']),p(conclusion['compute']),
            p('Four focused checks passed. Model and tokenizer bytes are verified before CPU model loading. Each regenerated row is chosen solely from question length or missing EOS in the prior model output; no public reference answers enter the procedure. Source: 65f74d50bd9f6d68da0a555929783d0273a7d266. Evidence: experiments/sequence-length-audit-v1 and experiments/length-cap-cpu-v1.','SmallText')]
    decoder_run = ROOT/'experiments/decoder-dropout-1k-v1'
    if (decoder_run/'conclusion.json').exists():
        conclusion=json.loads((decoder_run/'conclusion.json').read_text())
        story += [PageBreak(),p('Decoder-input dropout experiment','Title'),
            p('A 15.20-second local CPU diagnostic used the existing best checkpoint and 256 reserved training examples. With correct previous target tokens, number/operator-piece prediction accuracy was 85.35%; pairing each target with another reserved question reduced it to 37.70%. This shows substantial question dependence. Removing all preceding-code embeddings reduced the same accuracy to 0.05%, but the interventions differ in severity: these numbers do not measure causal contribution shares or establish that dropout will help.'),
            p('Test three matched continuations from the verified 72,000-step model: unchanged control, 10% and 20% whole-token decoder embedding dropout. Each adds 1,024 updates at batch 128, numeric-token loss weight 2 and peak learning rate 3e-5. Keep every supplied question and reference target unchanged. Only training-time decoder embeddings are randomly zeroed, with retained embeddings scaled to preserve their expectation. Preserve start/end/padding embeddings and use a separate random generator. Inference stays strict greedy with dropout disabled.'),
            p(conclusion['decision']),p(conclusion['compute']),
            p('Five unittest checks preceded launch, covering dropout and continuation behavior. Two runtime-finalization checks were subsequently rerun explicitly with pytest (seven dropout/continuation/runtime checks in total). Parent optimizer/data preflight was reused with checksum verification. Inspiration: Zhang et al., Token Drop mechanism for Neural Machine Translation, COLING 2020, https://aclanthology.org/2020.coling-main.379/. Only whole-token embedding dropout is tested; no extra targets, teacher or auxiliary objective is adopted. Source: a4a79fb19b7a45ba1bcec3517309d11cfe1d8b02. Evidence: experiments/conditioning-audit-cpu-v1 and experiments/decoder-dropout-1k-v1.','SmallText')]
    copy_run = ROOT/'experiments/copy-adapter-cpu-v1'
    if (copy_run/'conclusion.json').exists():
        conclusion=json.loads((copy_run/'conclusion.json').read_text())
        story += [PageBreak(),p('Learned source copying: CPU pilot','Title'),
            p('A 9.54-second CPU diagnostic on 256 reserved original examples found 207 number-piece prediction errors. The exact target token occurred in the question for 62 errors across 41 examples; an untrained average of all attention heads recovered 21 of those 62. These teacher-forced token diagnostics are not whole-program accuracy or a guaranteed achievable gain. Number pieces also include comments and strings.'),
            p('Keep all 42,481,152 parent Transformer parameters frozen. Fit 545 parameters: a decoder-state copying gate and a learned mixture of 32 cross-attention heads. At every step, mix normal vocabulary probabilities with probabilities of copying existing question token IDs. Sum repeated source tokens, exclude padding/end tokens, and take the strict greedy maximum. No program repair, hand-written answer rule, generated target or pretrained component is used.'),
            p('Use 4,096 original examples from supplied training shard zero, excluding all questions in the first 1,024 held-out rows. Fit for 512 small updates with the existing numeric/operator weight 2. Save and checksum-verify the custom CopyT5 checkpoint on Drive before generation, then reload the identical local weights; verify that every base parameter is unchanged. Compare copying enabled versus disabled on the same saved model and held-out subsets. Export all 1,919 test predictions in official order for bot evaluation.'),
            p(conclusion['decision']),p(conclusion['compute']),
            p('Eleven focused pytest checks passed, including cached greedy generation, save/reload, copying normalization, gradient flow, disabled equivalence and runtime finalization. Real-checkpoint CPU preflight confirmed finite adapter gradients and no base gradients. Source: 0938a6265db552fd794e8e691cb30d646069ea60. Evidence: experiments/copyability-audit-cpu-v1 and experiments/copy-adapter-cpu-v1. Pointer-generator reference: See, Liu and Manning (2017), https://aclanthology.org/P17-1099/. Reused validation is exploratory; no long run is automatic.','SmallText')]
    optimizer_run = ROOT/'experiments/optimizer-scratch-2k-v1'
    if (optimizer_run/'conclusion.json').exists():
        conclusion=json.loads((optimizer_run/'conclusion.json').read_text())
        r=json.loads((optimizer_run/'selection.json').read_text())
        rows=[['Optimizer / rate','Pilot /245','Confirm /493','Failures','Public /960']]
        for name,stages in r['results'].items():
            receipt=optimizer_run/name/'step-002048/bot_receipt.json'
            public=json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
            label={'control':'AdamW 0.0003','adamw_high':'AdamW 0.001','muon_low':'Muon 0.001','muon_high':'Muon 0.003'}[name]
            rows.append([label,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(v['prediction_failures'] for v in stages.values()),public])
        story += [PageBreak(),p('Full-model optimizer comparison from scratch','Title'),
            p('Four 42.48M Transformers start from identical random weights, with the same original 8k tokenizer, original example stream, seed 42, batch 128 and 2,048 updates. Every parameter trains. AdamW uses rates 0.0003 and 0.001. Muon uses rates 0.001 and 0.003 on hidden matrices; its embeddings and vectors use AdamW at 0.0003. Native PyTorch Muon uses five Newton-Schulz iterations, momentum 0.95 and match_rms_adamw scaling. Tied output/input embeddings and relative-position embeddings are excluded from Muon.'),
            table(rows,[46*mm,29*mm,35*mm,25*mm,35*mm]),
            p('Choose the stronger AdamW rate by summed held-out agreements, breaking ties by fewer failures then lower rate. Muon must improve both subsets without extra failures. Each survivor requires another seed before any longer or combined run. This compares early fresh-model learning; it does not test switching optimizers on the mature model, and it does not establish optimal rates or model size. Public feedback is recorded separately.'),
            p(conclusion['decision']),p(conclusion['compute']),
            p('Twelve focused CPU checks passed, plus four discarded full-size one-step CPU preflights with identical initial loss and finite gradients for every model parameter. Colab CPU confirmed native Muon before GPU allocation. A discarded GPU timing probe estimated 0.104 seconds/step for AdamW and 0.115 for Muon on the same 128-example batch. Source: bb3c316097764701dc0eaa822b3eafc5decaf9b1. Native implementation: https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html. Inspiration: https://github.com/KellerJordan/modded-nanogpt; its language-model speedrun results do not imply gains on this task.','SmallText')]
    optimizer_confirm = ROOT/'experiments/optimizer-confirm-2k-v1'
    if (optimizer_confirm/'conclusion.json').exists():
        conclusion=json.loads((optimizer_confirm/'conclusion.json').read_text())
        r=json.loads((optimizer_confirm/'selection.json').read_text())
        rows=[['Recipe','Pilot /245','Confirm /493','Failures','Public /960']]
        for name,stages in r['results'].items():
            receipt=optimizer_confirm/name/'step-002048/bot_receipt.json'
            public=json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
            rows.append([name,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(v['prediction_failures'] for v in stages.values()),public])
        story += [PageBreak(),p('Muon: second-seed replication','Title'),
            p('Repeat only the selected AdamW 0.0003 and Muon 0.001 recipes with shared seed 1042. The original screen used seed 42. Every parameter of each 42.48M encoder-decoder trains from scratch for 2,048 updates; the original 8k tokenizer, batch 128, original target loss, warmup and data pipeline stay fixed. Muon uses AdamW 0.0003 for embeddings and vectors. No frozen backbone, pretrained weights, generated targets or test feedback enters training.'),
            table(rows,[46*mm,29*mm,35*mm,25*mm,35*mm]),
            p('The candidate must improve both held-out subsets and add no total execution failures in both seeds. The prior selection file is checksum-verified before launch. These reused validation questions make the result exploratory; replication across seeds does not establish statistical significance, a compute-optimal model size or eventual mature-model accuracy. Equal examples also do not imply equal GPU cost.'),
            p(conclusion['decision']),p(conclusion['compute']),
            p('Ten focused checks passed. Existing full-size CPU model/optimizer checks and native-Muon Colab compatibility evidence were reused because the implementations and rates are unchanged. Preparation files were uploaded to Drive and downloaded back before allocation. Each checkpoint and strict-greedy test output was saved to Drive before export and exact-token verification. Source: 310a4d89ec8fd62d58d59e7042134cd4bd7c23ac. Evidence: experiments/optimizer-confirm-2k-v1.','SmallText')]
    depth_run = ROOT/'experiments/depth-muon-2k-v1'
    if (depth_run/'conclusion.json').exists():
        conclusion=json.loads((depth_run/'conclusion.json').read_text())
        r=json.loads((depth_run/'selection.json').read_text())
        rows=[['Encoder / decoder','Pilot /245','Confirm /493','Failures','Public /960']]
        for name,stages in r['results'].items():
            receipt=depth_run/name/'step-002048/bot_receipt.json'
            public=json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
            label={'control':'6 / 4 (42.48M)','encoder_heavy':'8 / 2 (42.35M)','decoder_heavy':'4 / 6 (42.61M)'}[name]
            rows.append([label,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(v['prediction_failures'] for v in stages.values()),public])
        story += [PageBreak(),p('Encoder/decoder capacity balance with Muon','Title'),
            p('Compare three fresh models using the replicated Muon 0.001 recipe, with AdamW 0.0003 on embeddings and vectors. All have model width 512 and eight attention heads. Encoder/decoder depths 6/4, 8/2 and 4/6 use feed-forward widths 1536, 1664 and 1408 respectively, keeping parameter counts within 0.32% of baseline. Each uses seed 42, the original 8k tokenizer and unmodified data, batch 128 and 2,048 updates. Different tensor shapes cannot share identical random weights; each initial model hash is recorded.'),
            table(rows,[46*mm,29*mm,35*mm,25*mm,35*mm]),
            p('A candidate must improve both held-out subsets with no extra total execution failures versus the newly trained 6/4 Muon control, then repeat that gain on another seed. This changes depth balance and compensating feed-forward width together. Equal parameters and examples do not imply equal compute; architecture-specific learning rates were not tuned. Public scores are recorded separately and do not select candidates.'),
            p(conclusion['decision']),p(conclusion['compute']),
            p('CPU preflight found that Transformers 4.57.6 inferred the default T5 cache depth from encoder num_layers, raising IndexError for the 4/6 model. Deeper-decoder models now save native dynamic_full caching in generation_config. Cached and uncached greedy IDs agree in the focused regression test, including after save/reload. Full-size finite-gradient and cached-generation checks pass for all three shapes; 13 focused checks passed before allocation. No GPU was used for this fix. Source: f6fc269034b3e42ec3e3e8850ecab63d1e726d76. Evidence: experiments/depth-muon-2k-v1.','SmallText')]
    depth_confirm = ROOT/'experiments/depth-confirm-2k-v1'
    if (depth_confirm/'conclusion.json').exists():
        conclusion=json.loads((depth_confirm/'conclusion.json').read_text())
        r=json.loads((depth_confirm/'selection.json').read_text())
        rows=[['Seed 1042 model','Pilot /245','Confirm /493','Failures','Public /960']]
        for name,stages in r['results'].items():
            receipt=json.loads((depth_confirm/name/'step-002048/bot_receipt.json').read_text())
            rows.append(['6 / 4 control' if name=='control' else '4 / 6 candidate',stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(v['prediction_failures'] for v in stages.values()),receipt['public_correct']])
        story += [PageBreak(),p('Second-seed depth confirmation','Title'),
            p('Repeat the 6/4 control versus 4/6 encoder/decoder candidate using seed 1042. The original tokenizer, examples, unmodified target programs, batch 128, 2,048 updates and Muon/AdamW rates remain fixed. All model parameters are trained from scratch. The original seed-42 selection is checksum-verified before the confirmation run. This repeats the same nearly matched-parameter architecture comparison; shapes and compensating feed-forward width differ.'),
            table(rows,[46*mm,29*mm,35*mm,25*mm,35*mm]),
            p(conclusion['decision']),p(conclusion['compute']),
            p('The candidate must improve both held-out subsets with no extra total execution failures in both seeds. The public bot score is logged separately. Reused validation makes this exploratory; replication does not establish eventual mature-model accuracy or statistical significance. Fifteen focused CPU checks passed before allocation; prior full-size finite-gradient and cache checks were reused because the model, optimizer, input and evaluation implementations are unchanged. Each checkpoint and exact greedy output was saved to Drive before export and bot submission. Source: 10520ebb061a9d7b4353bc1f1fa53664030b128b. Evidence: experiments/depth-confirm-2k-v1.','SmallText')]
    speed_comparison = ROOT/'experiments/muon-batch-probe-v1/comparison.json'
    if speed_comparison.exists():
        c=json.loads(speed_comparison.read_text())
        rows=[['GPU','Batch','Examples/s','Peak GiB','Units/hour']]
        for r in c['measurements']:
            rows.append([r['gpu'],r['batch_size'],round(r['examples_per_second']),round(r['peak_allocated_gib'],1),r['units_per_hour']])
        story += [PageBreak(),p('GPU and batch-size efficiency','Title'),
            p('A disposable Muon training probe uses the same scratch 42.48M model, original tokenizer, BF16, and encoded input path with two prefetched batches. Each timed measurement processes the same 8,192 original shard-0 training examples, excluding held-out questions. Each size receives three warm-up updates, followed by ascending and descending measurement rounds. Useful token counts are equal; padding differs. Results are medians of two short measurements, not full-dataset training speed or evidence of better learning.'),
            table(rows,[42*mm,20*mm,35*mm,35*mm,38*mm]),
            p(c['decision']),p(c['hardware_availability']),
            p('All probe weights are discarded. The supplied code, prepared inputs, hashes, logs and measured results are preserved on Drive. Each runtime has a bounded worker and automatic release. A batch change must separately pass a short quality comparison before adoption. CPU checks verified equal useful token counts before allocation. Benchmark source: commit 9e1a489. Evidence: experiments/muon-batch-probe-v1 and the adjacent H100/A100 attempt folders.','SmallText')]
    transfer = ROOT/'experiments/muon-transfer-1k-v1'
    if (transfer/'conclusion.json').exists():
        conclusion=json.loads((transfer/'conclusion.json').read_text())
        r=json.loads((transfer/'selection.json').read_text())
        rows=[['Optimizer arm','Pilot /245','Confirm /493','Failures','Public /960']]
        for name,stages in r['results'].items():
            receipt=transfer/name/'step-074048/bot_receipt.json'
            public=json.loads(receipt.read_text())['public_correct'] if receipt.exists() else 'pending'
            rows.append([name,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(v['prediction_failures'] for v in stages.values()),public])
        story += [PageBreak(),p('Muon transfer to the mature checkpoint','Title'),
            p('Four matched 1,024-update continuations start from the same own scratch-trained 73,024-step, 42.48M-parameter model (575/960 public). Compare AdamW with saved moments, reset AdamW, and reset Muon at matrix rates 0.0001 and 0.0003. AdamW on embeddings and vectors stays at 0.00003. Original tokenizer, examples, batch 128, numerical-token loss weight 2, 50-step warmup and cosine schedule remain matched. All parameters train. The reset AdamW arm separates the optimizer choice from discarding stored moments.'),
            table(rows,[46*mm,29*mm,35*mm,25*mm,35*mm]),
            p(conclusion['decision']),p(conclusion['compute']),
            p('Selection compares Muon against the stronger AdamW arm on the two reused held-out subsets. Require positive gains in both and no additional total execution failures; also require no loss against the parent (139/245 and 286/493). Any survivor requires a second-seed comparison. Public scores are recorded separately. This small screen cannot establish optimal rates or statistical significance.'),
            p('Eleven focused CPU tests pass, including optimizer-state reset, continuation schedules and selection gates. Full-size CPU forward/backward/update checks are finite for all four arms; the parent manifest is verified. Source f05c15878bb2be8bddf88f7385e48a580cbc84e7 and the parent archive were uploaded and downloaded back before GPU allocation. No pretrained weights, rewritten labels or test-reward optimization are used. Evidence: experiments/muon-transfer-1k-v1.','SmallText')]
    context_audit = ROOT/'experiments/context-length-cpu-v1/lengths.json'
    if context_audit.exists():
        audit=json.loads(context_audit.read_text())
        rows=[['Questions','Count','Mean tokens','Median','95th / 99th']]
        for name,label in [('train_shard_0','Eligible shard 0'),('official_test','Official test')]:
            s=audit['results'][name];q=s['quantiles']
            rows.append([label,s['rows'],f"{s['mean_tokens']:.2f}",q['0.5'],f"{q['0.95']} / {q['0.99']}"])
        cuts=[['Input cap','Train questions cut','Test questions cut']]
        for cap in ['64','96','128','192']:
            cuts.append([cap,*[f"{audit['results'][name]['caps'][cap]['questions_cut']:,} ({audit['results'][name]['caps'][cap]['questions_cut_percent']:.2f}%)" for name in ['train_shard_0','official_test']]])
        story += [PageBreak(),p('Question lengths and context limits','Title'),
            p('CPU-only audit performed after the encoder/decoder confirmation was scored and archived. Tokenize original questions with the current original 8k tokenizer, without padding or truncation; counts include the final end token. All eligible shard-0 rows exclude exact held-out question texts. This is not a census of the full 60-shard corpus. The earlier 15,304-question sample across all 60 shards averaged 46.86 tokens.'),
            table(rows,[46*mm,29*mm,35*mm,25*mm,35*mm]),
            table(cuts,[30*mm,70*mm,70*mm]),
            p('Retain the current 128-token input limit pending a quality and speed comparison. A 64-token limit would truncate about 12% of test questions. Training already groups similar lengths in windows of 32 batches, and both training and inference pad only to the longest sequence in each batch; a 128-token cap does not make every input 128 tokens long. These counts do not measure GPU savings or establish the best cap.'),
            p('The three test inputs above the current limit contain 137, 146 and 166 tokens including the end token. Input coverage is therefore a reason to test a larger limit as well as more efficient grouping. Neither the 128-token source cap nor the 192-token target cap changed. No model was trained, predictions generated or bot submission made for this audit. Original and mature-checkpoint tokenizer definitions agree after removing saved padding/truncation settings.'),
            p('Evidence: context_length_audit.py and experiments/context-length-cpu-v1. The CPU measurement took 3.29 seconds; no Colab allocation was made. Full histograms, input and tokenizer checksums, and the earlier sample reference are retained.','SmallText')]
    size_run = ROOT/'experiments/size-muon-2k-v1'
    if (size_run/'conclusion.json').exists():
        conclusion=json.loads((size_run/'conclusion.json').read_text())
        selected=json.loads((size_run/'selection.json').read_text())
        plan=json.loads((size_run/'plan.json').read_text())
        rows=[['Parameters','Pilot /245','Confirm /493','Failures','Public /960']]
        cost=[['Model','Process seconds','Val loss at 512 / 1024 / 1536 / 2048']]
        curves=json.loads((size_run/'learning-curves.json').read_text())
        for name,stages in selected['results'].items():
            receipt=json.loads((size_run/name/'step-002048/bot_receipt.json').read_text())
            budget=json.loads((size_run/name/'compute-budget.json').read_text())
            label=f"{plan['expected_parameters'][name]/1e6:.2f}M"
            rows.append([label,stages['pilot']['numeric_correct'],stages['confirmation']['numeric_correct'],sum(v['prediction_failures'] for v in stages.values()),receipt['public_correct']])
            losses=' / '.join(f"{v['token_loss']:.3f}" for v in curves[name])
            cost.append([label,f"{budget['process_seconds_including_setup']:.1f}",losses])
        story += [PageBreak(),p('Short model-size comparison with Muon','Title'),
            p('Three fresh encoder-decoder Transformers use the retained 6/4 depth and 64-dimensional attention heads. Model/feed-forward widths and head counts are 320/960/5, 512/1536/8 and 704/2112/11. All parameters train for 2,048 updates with seed 42, batch 128, BF16, the original 8k tokenizer and unmodified course examples. Hidden matrices use Muon 0.001; embeddings and vectors use AdamW 0.0003. Validation token loss is recorded every 512 updates, while the single final checkpoint per arm is saved and evaluated with strict greedy decoding.'),
            table(rows,[35*mm,30*mm,35*mm,30*mm,40*mm]),
            table(cost,[25*mm,35*mm,110*mm]),
            p(conclusion['decision']),p(conclusion['compute']),
            p('The predeclared gate requires improvement on both held-out subsets and no extra total execution failures versus the fresh 42M control. Public scores are recorded separately. A survivor still needs another seed and a comparison at matched measured GPU time before longer training. Equal updates do not equal compute; rates were not tuned separately for each size. These short curves and reused validation questions cannot establish an optimal final model size.'),
            p('Twelve focused CPU tests passed, plus finite-gradient and cached-versus-uncached greedy checks at all three actual model sizes. Files were uploaded to Drive and downloaded back before allocation. Every final checkpoint and generated output was saved to Drive before export, checksum verification and bot evaluation. Training source: 926d03d7834ef412e27c4a30bca173a978723c53. Evidence: experiments/size-muon-2k-v1.','SmallText')]
    doc = SimpleDocTemplate(str(OUTPUT),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,bottomMargin=22*mm,topMargin=20*mm,title='HSE NLP HW2 experiment report',author='Andrei Vorsin')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUTPUT)

if __name__ == '__main__':
    build()
