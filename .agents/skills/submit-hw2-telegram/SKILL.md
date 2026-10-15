---
name: submit-hw2-telegram
description: Prepare and submit this repository's HW2 predictions to @hse_nlp_hw2_2026_bot in Telegram, and record the bot's receipt and score. Use when asked to submit HW2 results or check a submission.
---

# Submit HW2 to Telegram

## Destination and files

- Bot: https://t.me/hse_nlp_hw2_2026_bot — verify the exact username before sending.
- Preferred chat in the Codex in-app browser: https://web.telegram.org/a/#8887250691 (display name: `HSE NLP Code Generation`).
- Assignment: https://github.com/ashaba1in/hse-nlp/blob/2026/homeworks/hw2_code_generation.md (local copy: `homeworks/hw2_code_generation.md`).
- Default candidate: `homeworks/hw2/predictions.csv`.
- Generator: `homeworks/hw2/inference.py`; Colab notebook: `homeworks/hw2/hw2_inference_colab.ipynb`.
- Experiment evidence: `homeworks/hw2/metrics.json` and saved notebook outputs.

Paths are relative to the repository root. Existing files can belong to an older experiment: identify which model/run produced the selected predictions before submitting.

## Prepare

1. Read the current assignment and the bot's visible instructions. Verified from its `/start` reply on 2026-10-15: it accepts **only `.jsonl`**, with one JSON object `{"code": "<model output>"}` per task, **1,919 tasks in the order of `pass1_questions.jsonl`**. It recommends generation of at most 64 tokens. Recheck for changed instructions rather than treating this date's contract as permanent.
2. The local generator writes CSV, which must be converted before uploading. Parse it with `csv.DictReader`, since programs can contain embedded newlines. Write `homeworks/hw2/predictions.jsonl` in UTF-8 using `json.dumps({"code": row["code"]}, ensure_ascii=False) + "\n"` for each row. Do not manually escape code, truncate existing answers to 64 tokens, sort or deduplicate rows. Read the JSONL back and verify every code string equals its CSV source and is nonempty.
3. Verify the count against the actual test split and establish that its question order matches `pass1_questions.jsonl`. Row count alone does not prove alignment. If that correspondence cannot be established, report the missing evidence before submitting; do not assume the CSV is aligned merely because it has 1,919 rows.
4. Record the uploaded JSONL path, SHA-256, source commit and model/run identity. Check whether generation on one NVIDIA T4 met the assignment's 180-second limit. A faster GPU's timing is not evidence for T4 compliance. Report missing evidence; do not describe an unchecked run as compliant.

## Send when requested

Creating this skill or preparing a file does not itself authorize sending. A user's explicit request to submit HW2 to this bot authorizes the ordinary submission steps; do not ask again for the same action. If the candidate is ambiguous, resolve it before uploading.

Use the **Codex in-app browser** through `cua_repl`, as explicitly selected by the user. On 2026-10-15 Telegram Web was already signed in there and the course bot answered `/start`. Reuse the existing chat tab when available; otherwise open the preferred chat URL with `cua.createBrowserTab("iab", url, {visible: true})`. Verify the current account/session and bot identity before sending. If the direct chat is unavailable, open `https://web.telegram.org/`, search the exact username `hse_nlp_hw2_2026_bot`, and open `HSE NLP Code Generation`. If sign-in or required registration information is unavailable, ask for that specific missing step.

Chrome's `t.me` page was blocked by Cold Turkey; the user subsequently explicitly requested the Codex browser, where Telegram Web worked. Prefer that established route. Do not disable Cold Turkey. If the selected route becomes blocked too, report the blocker.

Read the existing conversation first. Use Start or help only when needed and supported by the visible bot UI. For upload:

1. Read the browser's `file-uploads` documentation before the first upload and inspect fresh page state.
2. Click the chat's **Add an attachment** control (observed ID: `attach-menu-button`), then the file/document option shown by the current menu.
3. Use the documented file chooser API to select the absolute path to `homeworks/hw2/predictions.jsonl`. Do not paste the programs into the message box or attach the source CSV.
4. Inspect the attachment preview, confirm its filename and the recipient, then send once. Preserve the chat tab with `markDeliverable()` for a completed result or `markHandoff()` if waiting for a response or user input.

The attachment control was observed, but uploading a file has **not yet been verified end to end**. Follow the live UI rather than assuming later menu labels or successful delivery. Send the report, model or other files only if the bot's instructions and the user's request call for them.

## Why use the user's account

Do not create another bot or request the course bot's token for this workflow. Direct bot-to-bot private messaging requires **Bot-to-Bot Communication Mode** enabled for both bots by their owners; the course bot's setting and acceptance of bot-owned submissions are unknown. Even if enabled, participant attribution needs verification. The established workflow submits through the user's Telegram account. Reference: https://core.telegram.org/api/bots/bot-to-bot.

## Verify and retain the result

- Inspect the bot's response after upload. Distinguish file sent, submission accepted, queued for evaluation and scored. An outgoing attachment alone does not prove acceptance.
- On a timeout or unclear response, inspect the chat for the existing submission before retrying. Avoid duplicate uploads. Respect the bot's rate limits; stop after a rejection and address the stated cause before resubmitting.
- Report the exact score and submission ID when provided. Do not infer pass@1 from syntax validity, token loss or exact-match diagnostics, and do not claim a private leaderboard result unless shown.
- Save a concise record in `homeworks/hw2/submissions/<timestamp>.json`: submission time with timezone, bot username, filename/hash, source commit, model/run, acceptance status, submission ID and reported score. Use null for unknown values. Keep Telegram credentials, session files and unrelated chat content out of the repository.
- Submission to this bot is separate from delivering the required source code, inference notebook and PDF report. State what was actually accepted and what remains to submit.
