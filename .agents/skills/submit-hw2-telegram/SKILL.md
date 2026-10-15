---
name: submit-hw2-telegram
description: Prepare and submit this repository's HW2 predictions to @hse_nlp_hw2_2026_bot in Telegram, and record the bot's receipt and score. Use when asked to submit HW2 results or check a submission.
---

# Submit HW2 to Telegram

## Destination and files

- Bot: https://t.me/hse_nlp_hw2_2026_bot — verify the exact username before sending.
- Assignment: https://github.com/ashaba1in/hse-nlp/blob/2026/homeworks/hw2_code_generation.md (local copy: `homeworks/hw2_code_generation.md`).
- Default candidate: `homeworks/hw2/predictions.csv`.
- Generator: `homeworks/hw2/inference.py`; Colab notebook: `homeworks/hw2/hw2_inference_colab.ipynb`.
- Experiment evidence: `homeworks/hw2/metrics.json` and saved notebook outputs.

Paths are relative to the repository root. Existing files can belong to an older experiment: identify which model/run produced the selected predictions before submitting.

## Prepare

1. Read the current assignment and the bot's visible instructions. The local generator currently writes UTF-8 CSV with one `code` column in test-set order. This is the repository's format, not a verified bot contract. Follow the bot's actual schema, filename, registration steps and submission limits; do not guess commands or required identity details.
2. Parse the candidate with a CSV reader, since programs can contain embedded newlines. Verify its columns, nonempty answers and row count against the actual test split used by the run. Check ordering from generation code or retained IDs; row count alone does not prove alignment. Do not sort or deduplicate predictions.
3. Record the candidate path, SHA-256, source commit and model/run identity. Check whether generation on one NVIDIA T4 met the assignment's 180-second limit. A faster GPU's timing is not evidence for T4 compliance. Report missing evidence; do not describe an unchecked run as compliant.

## Send when requested

Creating this skill or preparing a file does not itself authorize sending. A user's explicit request to submit HW2 to this bot authorizes the ordinary submission steps; do not ask again for the same action. If the candidate is ambiguous, resolve it before uploading.

Use an available Telegram connector with the user's account, or the signed-in Telegram app/web UI through computer-use tools. Do not invent a Telegram MCP tool or request a token for the course bot: it belongs to the course, and submissions are sent as the user. If sign-in or required registration information is unavailable, ask for that specific missing step.

Read the existing conversation first. Use Start or help only when needed and supported by the visible bot UI. Attach the selected predictions as a document/file, verify the attachment name and recipient, then send once. Send the report, model or other files only if the bot's instructions and the user's request call for them.

## Verify and retain the result

- Inspect the bot's response after upload. Distinguish file sent, submission accepted, queued for evaluation and scored. An outgoing attachment alone does not prove acceptance.
- On a timeout or unclear response, inspect the chat for the existing submission before retrying. Avoid duplicate uploads. Respect the bot's rate limits; stop after a rejection and address the stated cause before resubmitting.
- Report the exact score and submission ID when provided. Do not infer pass@1 from syntax validity, token loss or exact-match diagnostics, and do not claim a private leaderboard result unless shown.
- Save a concise record in `homeworks/hw2/submissions/<timestamp>.json`: submission time with timezone, bot username, filename/hash, source commit, model/run, acceptance status, submission ID and reported score. Use null for unknown values. Keep Telegram credentials, session files and unrelated chat content out of the repository.
- Submission to this bot is separate from delivering the required source code, inference notebook and PDF report. State what was actually accepted and what remains to submit.
