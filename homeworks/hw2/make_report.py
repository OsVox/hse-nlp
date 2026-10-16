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
    t4 = json.loads(t4_path.read_text()) if t4_path.exists() else None
    story = [p('Code generation from scratch','Title'),
        p('HW2 experiment report - work in progress<br/>Andrei Vorsin | 16 October 2026','SmallText'),
        p('Current result','Head'),
        p(f"Verified and scored checkpoints reach {extended[-1]['step']:,} updates. Ten initial checkpoints, eight first-continuation checkpoints and {len(extended)} longer-continuation checkpoints were saved in Google Drive and scored by the course bot. The best model-only public result is <b>{best['public_percent']:.2f}% ({best['correct']}/960)</b> at step {best['step']:,}, rank {best['rank']} of {best['participants']} at {best['time']}. The 33% baseline is exceeded; first place remains unmet. No private result is available."),
        p('Task and rules','Head'),
        p('Train an encoder-decoder Transformer from random weights using only the supplied SlavaYus/tinysms_qa_12m training split. Generate one Python program per test question. All 1,919 programs must be generated within 180 seconds on one NVIDIA T4. No pretrained weights, outside examples, model distillation, or agent-written target programs are used.'),
        p('Data and architecture','Head'),
        p('The allowed dataset contains 11,838,003 question/code pairs in 60 training files. The test file contains questions only. A byte-level BPE tokenizer with 8,192 tokens was fitted on shard 0 after reserving its first 1,024 rows. The saved model has 9,967,360 parameters: four encoder layers, four decoder layers, width 256, feed-forward width 768, four attention heads, gated GELU and dropout 0.1. It uses the T5 architecture with randomly initialized weights.'),
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
        p('Generation sorts questions by token length and restores their original positions. Greedy decoding is followed by four-beam generation for syntactically invalid outputs; the first parseable model candidate is retained. No candidate is rewritten. Selected token IDs are saved, then decoded again to check equality with every output string. A sidecar records source commit, checkpoint hash, question-order hash, output hash, device and generation settings. JSONL conversion only wraps each exact string as a code field.'),
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
        p('Prepare and verify the final step-180000 model, tokenizer, official 1,919 questions and source on CPU. Save a hash-verified bundle to Drive, export and verify its ready record, then allocate one T4. Compare FP32 batch sizes 64, 128 and 256, twice with reversed order. Keep greedy generation, cached decoder state, length sorting, syntax-only beam retry and input/output limits 128/192 unchanged. Save every exact output, token provenance, hash and measurement to Drive. Use a separate profiling pass; exclude its synchronization overhead from the timing comparison.'),
        p('Generation time includes tokenization, sorting, generation, retries, decoding and local CSV writing. Full-call time additionally covers test reading, model hashing, provenance writing and verification. Model load and a small warm-up are measured separately. Drive archival is outside the generation timer. The experiment has a 600-second soft budget and a bounded child-process timeout; the notebook requests GPU release in finally.'),
    ]
    if t4 and t4.get('completed'):
        rows = [['Batch','Median generation, s','Slowest run, s','Same outputs as 64']]
        for item in t4['summary']:
            rows.append([item['batch_size'],f"{item['median_seconds']:.3f}",
                         f"{item['max_seconds']:.3f}",item['all_outputs_equal_baseline']])
        story += [table(rows,[30*mm,50*mm,45*mm,45*mm]),
                  p(f"Device: {escape(t4['gpu'])}; PyTorch {escape(t4['torch'])}; Transformers {escape(t4['transformers'])}. Model loading took {t4['model_load_seconds']:.3f} seconds and warm-up took {t4['warmup_seconds']:.3f} seconds. Changed programs require separate quality evaluation; byte-identical outputs preserve the same predictions."),
                  p('Full runs, per-call times, memory peaks, provenance and output hashes are retained in experiments/inference-speed-180k-t4-v1/results.json. Reproduce with hw2_inference_benchmark_colab.ipynb. The requirement applies to measured generation, not to an extrapolation from Blackwell.')]
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
            p('These are early-learning results from one seed, not final convergence estimates. Even the best pilot remains below the established 180k model at 53.85%. The bot reported public rank 10/15 on 16 October; earlier rank 9/15 was an earlier snapshot. No private score or actual T4 timing is available for these pilots.'),
            p('Time and next comparison', 'Head'),
            p('The driver took 718.51 seconds after data restoration; its launcher record spans 758.79 seconds including restoration and imports. Across 716 GPU samples, mean activity was 72.90%, median 97%, and peak memory 18,627 MiB. Activity is not achieved compute throughput; startup, I/O and execution evaluation contribute low samples. Batch-size speed still needs a matched benchmark.'),
            p('Continue both finalists from 2k to 6k with their optimizer moments, batch 128 and respective learning rates. Both receive the same new deterministic sample stream, a restarted 4k cosine schedule and 200 warm-up updates. Save and score the 6k checkpoint for each. Check actual T4 inference before long training. The four-way GPU was released manually after the automatic notebook request left it connected; file work resumed on CPU.'),
            p('Evidence: experiments/short-pilots-v1, its selection.md and export.json, four submission receipts, finalist_pilots.py and hw2_finalist_pilots_colab.ipynb. Full weights, optimizer states and exact predictions remain in Google Drive.', 'SmallText')]
    doc = SimpleDocTemplate(str(OUTPUT),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,topMargin=20*mm,bottomMargin=22*mm,title='HSE NLP HW2 experiment report',author='Andrei Vorsin')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUTPUT)

if __name__ == '__main__':
    build()
