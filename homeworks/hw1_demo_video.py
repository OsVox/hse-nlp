"""Render an illustrated walkthrough from verified live widget interactions.

The MP4 is an illustration of the tested notebook UI, not a screen recording.
Run `python homeworks/hw1_demo_video.py` to recreate it with Pillow and ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS, SECONDS = 960, 540, 10, 68
OUTPUT = Path(__file__).with_name("hw1_demo.mp4")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


TITLE = font(29, bold=True)
BODY = font(20)
SMALL = font(16)
BUTTON = font(19)


def center(draw: ImageDraw.ImageDraw, y: int, message: str, face, color: str) -> None:
    box = draw.textbbox((0, 0), message, font=face)
    draw.text(((WIDTH - box[2]) / 2, y), message, fill=color, font=face)


def render(second: float) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#f4f6fa")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 63), fill="#273b56")
    draw.text((36, 18), "NLP homework 1", fill="white", font=font(23, bold=True))
    draw.text((735, 22), "Text suggestion", fill="#c7e4ff", font=SMALL)

    if second < 6:
        center(draw, 154, "Type faster with word suggestions", TITLE, "#24344b")
        center(draw, 211, "A prefix tree completes words; an n-gram model continues text.", BODY, "#4d5e72")
        center(draw, 443, "Illustrated walkthrough from a tested Jupyter widget", SMALL, "#718096")
        return canvas

    if second < 30:
        prompt = "Please let me kn"
        typed = prompt[:min(len(prompt), int((second - 6) * 1.7))]
        suggestions = ["know if you have", "know if you are", "know if you would"] if second >= 17 else []
        note = "Suggestions update as the word is typed."
        selected = 0 if second >= 25 else -1
    elif second < 43:
        typed = "Please let me know if you have "
        suggestions = []
        note = "One click replaces the partial word and inserts the continuation."
        selected = -1
    elif second < 56:
        prompt = "Thank you for your "
        typed = prompt[:min(len(prompt), int((second - 43) * 2))]
        suggestions = ["help in this issue"] if second >= 50 else []
        note = "After a space, the model uses the preceding words."
        selected = 0 if second >= 54 else -1
    else:
        typed = "Thank you for your help in this issue "
        suggestions = []
        note = "The chosen suggestion is inserted into the email."
        selected = -1

    draw.rounded_rectangle((75, 103, 885, 305), radius=12, fill="white", outline="#ced7e4", width=2)
    draw.text((99, 123), "Email", fill="#556579", font=SMALL)
    draw.text((99, 171), typed + ("|" if int(second * 2) % 2 == 0 else ""), fill="#1c2d43", font=BODY)
    draw.text((75, 330), "Suggestions - click one to insert it", fill="#34455c", font=font(17, bold=True))
    for index in range(3):
        top = 365 + index * 43
        active = index == selected
        draw.rounded_rectangle(
            (75, top, 885, top + 36), radius=6,
            fill="#d6e9ff" if active else "#e8edf4",
            outline="#6ba5e7" if active else "#e8edf4", width=2,
        )
        if index < len(suggestions):
            center(draw, top + 7, suggestions[index], BUTTON, "#16385d")
    draw.text((75, 503), note, fill="#50627a", font=SMALL)
    return canvas


def main() -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUTPUT),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        for frame in range(FPS * SECONDS):
            encoder.stdin.write(render(frame / FPS).tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg could not render the walkthrough")
    print(OUTPUT)


if __name__ == "__main__":
    main()
