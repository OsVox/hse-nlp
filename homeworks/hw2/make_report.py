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
    canvas.drawString(20*mm,13*mm,'HSE NLP HW2 | Andrei Vorsin | 15 October 2026')
    canvas.drawRightString(190*mm,13*mm,str(doc.page))
    canvas.restoreState()


def build():
    metrics = json.loads((ROOT/'metrics.json').read_text())
    submission = json.loads((ROOT/'submissions/2026-10-15-gpu-check.json').read_text())
    quality_path = ROOT/'quality_results.json'
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
    current = json.loads((ROOT/'experiments/mixed-lr001-20k-drive-v1/summary.json').read_text())
    best = max(current['checkpoints'], key=lambda x: x['public_pass_at_1_percent'])
    story = [p('Code generation from scratch','Title'),
        p('HW2 experiment report - work in progress<br/>Andrei Vorsin | 15 October 2026','SmallText'),
        p('Current result','Head'),
        p(f"The completed 20,000-update run has ten checkpoints saved in Google Drive and scored by the course bot. Its best model-only public result is <b>{best['public_pass_at_1_percent']:.2f}% ({best['correct']}/960)</b> at step {best['step']:,}. The 33% baseline and first-place goal remain unmet. No private leaderboard result is available."),
        p('Task and rules','Head'),
        p('Train an encoder-decoder Transformer from random weights using only the supplied SlavaYus/tinysms_qa_12m training split. Generate one Python program per test question. All 1,919 programs must be generated within 180 seconds on one NVIDIA T4. No pretrained weights, outside examples, model distillation, or agent-written target programs are used.'),
        p('Data and architecture','Head'),
        p('The allowed dataset contains 11,838,003 question/code pairs in 60 training files. The test file contains questions only. A byte-level BPE tokenizer with 8,192 tokens was fitted on shard 0 after reserving its first 1,024 rows. The saved model has 9,967,360 parameters: four encoder layers, four decoder layers, width 256, feed-forward width 768, four attention heads, gated GELU and dropout 0.1. It uses the T5 architecture with randomly initialized weights.'),
        p('The current run uses AdamW, learning rate 0.001 (historical runs used 0.0003), 500 warmup updates, cosine decay to 10% of the initial rate, weight decay 0.01 and gradient clipping at 1.0. Input and target limits are 128 and 192 tokens. Padding tokens do not contribute to the loss. Model state and optimizer state are saved at regular intervals.'),
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
        p('The historical 39.09-second T4 measurement belongs to an older run. Blackwell timings cannot establish the T4 requirement for the current candidate. The inference notebook now checks the actual GPU name and rejects a non-T4 speed verification. The final checkpoint and decoding configuration require a fresh complete T4 measurement before compliance is claimed.'),
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
        p('The early pilot weights were lost when temporary runtime storage was released before export succeeded; their measurements remain saved. The revised workflow passed CPU save and restore checks, then completed a fresh run with all ten checkpoints and their exact predictions retained in Drive. Every checkpoint has a bot receipt. The GPU has been released.'),
        p('The public baseline and first-place objective remain unmet. Before selecting a final model, compare held-out numeric results, preserve its checkpoint and training evidence, run full generation on one T4, and obtain an actual score from the course bot. A public first-place result would not establish private first place. No private result is claimed.'),
        p('Reproduction files','Head'),
        p('Repository: <link href="https://github.com/OsVox/hse-nlp/tree/2026/homeworks/hw2">OsVox/hse-nlp, branch 2026, homeworks/hw2</link>. The directory contains train.py, data.py, numeric_eval.py, inference.py, tokenizer files, saved model weights, inference and training notebooks, experiment records and submission receipts. quality_experiments.md specifies the controlled pilot before its run. The raw logs and run configurations accompany the quality results.'),
        p('Use a fresh output directory for each experiment. Run training with the recorded settings and keep latest plus best checkpoints. Run numeric evaluation on reserved training rows only. Use hw2_inference_colab.ipynb on a T4 to generate and package the test outputs. Verify every checkpoint in Drive before releasing compute; export predictions and record bot receipts from Drive.'),
        p('Sources','Head'),
        p('<link href="https://github.com/ashaba1in/hse-nlp/blob/2026/homeworks/hw2_code_generation.md">Course assignment</link>; <link href="https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m">allowed dataset and schema</link>; course week 2 (encoder-decoder attention and decoding), week 3 (training methods), week 4 (attention and caching). Local measurements: metrics.json, gpu_benchmark/*.json, quality_results.json when present, and submissions/2026-10-15-gpu-check.json.','SmallText'),
    ]
    doc = SimpleDocTemplate(str(OUTPUT),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,topMargin=20*mm,bottomMargin=22*mm,title='HSE NLP HW2 experiment report',author='Andrei Vorsin')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUTPUT)

if __name__ == '__main__':
    build()
