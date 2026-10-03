"""Build the PDF experiment report from checked metrics.json values."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Circle, Drawing, Line, String
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).parent
METRICS = json.loads((ROOT / "metrics.json").read_text())
OUTPUT = ROOT / "report.pdf"


def font_path(bold: bool = False) -> str:
    options = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in options:
        if Path(path).exists():
            return path
    raise FileNotFoundError("A Cyrillic TrueType font is required")


pdfmetrics.registerFont(TTFont("Report", font_path()))
pdfmetrics.registerFont(TTFont("ReportBold", font_path(True)))
pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold")

INK = colors.HexColor("#22334d")
MUTED = colors.HexColor("#55667b")
BLUE = colors.HexColor("#356aa5")
ORANGE = colors.HexColor("#cf7c35")
PALE = colors.HexColor("#eef4fb")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", fontName="ReportBold", fontSize=19, leading=23,
    textColor=INK, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="ReportSubtitle", fontName="Report", fontSize=9.5, leading=14,
    textColor=MUTED, spaceAfter=14,
))
styles.add(ParagraphStyle(
    name="ReportHead", fontName="ReportBold", fontSize=11.5, leading=15,
    textColor=INK, spaceBefore=12, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="ReportBody", fontName="Report", fontSize=9.2, leading=13.5,
    textColor=INK, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="ReportSmall", fontName="Report", fontSize=8, leading=11.5,
    textColor=MUTED, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="ReportCell", fontName="Report", fontSize=8.4, leading=11,
    textColor=INK,
))
styles.add(ParagraphStyle(
    name="ReportCellBold", fontName="ReportBold", fontSize=8.4, leading=11,
    textColor=INK,
))
styles.add(ParagraphStyle(
    name="ReportFooter", fontName="Report", fontSize=8, leading=10,
    textColor=MUTED, alignment=TA_CENTER,
))


def para(text: str, style: str = "ReportBody") -> Paragraph:
    return Paragraph(text, styles[style])


def page(canvas, document):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#d7e2ef"))
    canvas.line(19 * mm, height - 17 * mm, width - 19 * mm, height - 17 * mm)
    canvas.setFont("Report", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(19 * mm, 13 * mm, "Домашняя работа 2 · NLP")
    canvas.drawRightString(width - 19 * mm, 13 * mm, f"{document.page}")
    canvas.restoreState()


def metric(value, suffix: str = "", decimals: int = 4) -> str:
    if value is None:
        return "не измерено"
    if isinstance(value, float):
        return f"{value:.{decimals}f}{suffix}"
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ") + suffix
    return f"{value}{suffix}"


def loss_chart(series: list[tuple[str, list[dict], colors.Color]]) -> Drawing:
    chart = Drawing(155 * mm, 54 * mm)
    left, bottom, width, height = 15 * mm, 9 * mm, 132 * mm, 37 * mm
    chart.add(Line(left, bottom, left + width, bottom, strokeColor=MUTED, strokeWidth=0.7))
    chart.add(Line(left, bottom, left, bottom + height, strokeColor=MUTED, strokeWidth=0.7))
    maximum_step = max(point["step"] for _, observations, _ in series for point in observations)
    y_max = max(4.5, max(point["token_loss"] for _, observations, _ in series for point in observations) + 0.2)
    for tick in (0, 1, 2, 3, 4):
        y = bottom + tick / y_max * height
        chart.add(Line(left - 2 * mm, y, left + width, y,
                       strokeColor=colors.HexColor("#e5ebf2"), strokeWidth=0.5))
        chart.add(String(left - 5 * mm, y - 1.2 * mm, str(tick), fontName="Report",
                         fontSize=7, textAnchor="end", fillColor=MUTED))
    for fraction in (0, 0.5, 1):
        x = left + fraction * width
        chart.add(String(x, bottom - 4 * mm, f"{int(maximum_step * fraction):,}",
                         fontName="Report", fontSize=7, textAnchor="middle", fillColor=MUTED))
    for label, observations, color in series:
        points = sorted(observations, key=lambda point: point["step"])
        coords = [
            (left + point["step"] / maximum_step * width,
             bottom + point["token_loss"] / y_max * height)
            for point in points
        ]
        for first, second in zip(coords, coords[1:]):
            chart.add(Line(*first, *second, strokeColor=color, strokeWidth=2))
        for x, y in coords:
            chart.add(Circle(x, y, 2.2, fillColor=color, strokeColor=colors.white))
    chart.add(String(left, bottom + height + 4 * mm, "Отложенный token loss по шагам",
                     fontName="ReportBold", fontSize=9, fillColor=INK))
    for index, (label, _, color) in enumerate(series):
        x = left + 73 * mm + index * 30 * mm
        chart.add(Line(x, bottom + height + 5 * mm, x + 5 * mm,
                       bottom + height + 5 * mm, strokeColor=color, strokeWidth=2))
        chart.add(String(x + 6 * mm, bottom + height + 4 * mm, label,
                         fontName="Report", fontSize=7, fillColor=MUTED))
    return chart


def build() -> None:
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=19 * mm,
        rightMargin=19 * mm, topMargin=23 * mm, bottomMargin=21 * mm,
        title="Домашняя работа 2: генерация кода",
        author="Андрей Ворсин",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=page))

    story = [
        para("Домашняя работа 2: генерация кода", "ReportTitle"),
        para("Андрей Ворсин · ФКН ВШЭ · 3 октября 2026 года · " +
             ("эксперимент продолжается" if METRICS["run_status"] != "complete" else
              "итоговый отчет"), "ReportSubtitle"),
        para("Задача и ограничения", "ReportHead"),
        para(
            "Обучить с нуля Transformer с энкодером вопроса и декодером Python-кода. "
            "Данные - только <link href='https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m' "
            "color='#356aa5'>выданный датасет</link>: 11 838 003 пары question/code в train и "
            "1 919 вопросов без ответов в test. Для итогового решения требуется сгенерировать "
            "все тестовые ответы не дольше 180 секунд на одной NVIDIA T4. Публичный pass@1 "
            "оценивает проверяющая система курса; тестовые ответы не опубликованы."
        ),
        para("Данные и модель", "ReportHead"),
        para(
            "Для проверки отложены первые 1 024 пары первого train-шарда. Эти пары не участвуют "
            "в обучении модели и токенизатора. Токенизатор byte-level BPE размера 8 192 обучен "
            "на остальных парах того же шарда. Модель T5-типа создана со случайными весами: "
            f"{metric(METRICS['parameters'])} параметров, 4 слоя энкодера, 4 слоя декодера, "
            "размер скрытого состояния 256, 4 головы внимания. Ограничения длины: 128 токенов "
            "для вопроса и 192 для кода. Предобученные веса, дополнительные данные и искусственные "
            "таргеты не применялись."
        ),
        para("Обучение", "ReportHead"),
        para(
            f"AdamW, скорость обучения 3 × 10<super>-4</super>, warmup 500 обновлений, "
            f"batch {METRICS['batch_size']} × накопление градиента "
            f"{METRICS['gradient_accumulation']} = 64 примера на обновление; "
            "смешанная точность FP16, обрезка нормы градиента 1.0. "
            f"Пробег - {metric(METRICS['target_updates'])} обновлений на "
            f"{METRICS['device']}. Модель и состояние оптимизатора сохраняются каждые 500 шагов."
        ),
        para("Проверенные эксперименты", "ReportHead"),
        para(
            "Сначала один локальный шаг на Apple MPS проверил прохождение данных, обратный проход "
            "и сохранение модели (train loss 9.6595; held-out token loss 9.0907). "
            "Затем 10 шагов на T4 проверили совместимость среды и сохранение чекпоинта. "
            f"Пилотный T4-прогон дошел до {metric(METRICS['t4_pilot_update'])} шагов "
            f"(loss {metric(METRICS['t4_pilot_validation_loss'])}), но связь прервалась и "
            "итоговый чекпоинт не был получен. Этот пилот не участвует в выборе итоговой модели."
        ),
        para(
            "На корпоративной G4 проведены два запуска с одинаковой архитектурой, бюджетом "
            "обновлений и гиперпараметрами. Бейзлайн читает каждый перемешанный шард целиком; "
            "второй вариант переходит к следующему шарду после 1 000 мини-батчей. "
            "Гипотеза: при том же числе обновлений большее покрытие train-шардов улучшит "
            "обобщение на отложенных вопросах. "
            f"Итоговый validation loss: {metric(METRICS['baseline_g4_final_loss'])} "
            f"против {metric(METRICS['rotated_g4_final_loss'])}. "
            f"У бейзлайна был лучший записанный loss "
            f"{metric(METRICS['baseline_best_logged_loss'])} на шаге "
            f"{metric(METRICS['baseline_best_logged_step'])}, однако его веса не сохранились: "
            "остался только последний чекпоинт. "
            f"Для ответов выбрана модель: {metric(METRICS['selected_experiment'])}. "
            f"Чекпоинт выбран по минимальному loss на первых 256 отложенных парах "
            f"(шаг {metric(METRICS.get('selected_checkpoint_step'))}, "
            f"loss {metric(METRICS.get('selected_checkpoint_loss'))}). "
            "Гипотеза о ротации поддержана меньшим минимальным loss; на отдельных 256 "
            "парах точных текстовых совпадений было 2 против 0 у бейзлайна."
        ),
    ]

    observations = METRICS["validation"]
    selected_steps = {500, 1000, 3000, 5000, 10000, METRICS["selected_checkpoint_step"], 20000}
    selected = [x for x in observations if x["step"] in selected_steps]
    rows = [[para("Шаг", "ReportCellBold"), para("Loss на отложенных токенах", "ReportCellBold")]]
    rows += [[para(metric(item['step']), "ReportCell"),
              para(f"{item['token_loss']:.4f}", "ReportCell")]
             for item in selected]
    table = Table(rows, colWidths=[38 * mm, 86 * mm], hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d7e2ef")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.extend([
        Spacer(1, 4 * mm),
        loss_chart([
            ("Бейзлайн", METRICS["baseline_validation"], BLUE),
            *([("Ротация", METRICS["rotated_validation"], ORANGE)]
              if METRICS.get("rotated_validation") else []),
        ]),
        Spacer(1, 5 * mm),
        para(
            "Token loss измеряет предсказание следующего токена при известном правильном префиксе. "
            "Это не pass@1 и не проверка корректности выполненной программы.", "ReportSmall"
        ),
        para("Генерация и результат", "ReportHead"),
        para(
            "Основная генерация жадная, пакетами по 64 вопроса. Вопросы сортируются "
            "по длине для уменьшения padding, затем ответы возвращаются в исходный порядок. "
            "Только синтаксически неверные ответы повторно генерируются с четырьмя лучами; "
            "один разделенный пробелом идентификатор исправлен, и результат проверен Python-парсером. "
            "Время измеряется на T4 с синхронизацией CUDA вокруг генерации всех тестовых строк, "
            "включая сортировку, токенизацию, повторные попытки и запись CSV. "
            f"Результат: {metric(METRICS['test_predictions'], ' ответов')}; "
            f"время {metric(METRICS['t4_generation_seconds'], ' с', decimals=2)}. "
            "Все сгенерированные ответы непустые и синтаксически корректные.",
        ),
        para(
            "На 256 отложенных парах, не использованных для выбора чекпоинта, считаются "
            "доля непустого синтаксически корректного кода "
            f"({metric(METRICS['syntax_valid_fraction'])}), доля программ с функцией solve "
            f"({metric(METRICS['solve_defined_fraction'])}) и точное совпадение с эталоном "
            f"({metric(METRICS['exact_match_fraction'])}). Эти числа - диагностика, "
            "не замена pass@1: правильную задачу можно решить разным кодом. "
            f"Публичный pass@1: {metric(METRICS['public_pass_at_1'])}."
        ),
        KeepTogether([para("Использование агента", "ReportHead"), para(
            "Codex помог прочитать условие и схему датасета, написать и проверить код обработки "
            "данных, архитектуру, цикл обучения, инференс и отчет. Он нашел ошибку передачи "
            "token_type_ids в generate во время локальной проверки; ошибка исправлена до GPU-прогона. "
            "Агент не создавал обучающие пары, дополнительные таргеты и эталонные ответы и не "
            "использовал чужие веса. Итоговые численные результаты взяты из выполненных запусков."
        )]),
        para("Ограничения и воспроизводимость", "ReportHead"),
        para(
            "Формат посылки для Telegram-бота на момент работы еще не опубликован в условии. "
            "Код, токенизатор, Colab-ноутбуки и результаты лежат в форке "
            "<link href='https://github.com/OsVox/hse-nlp' color='#356aa5'>OsVox/hse-nlp</link>. "
            "Для повторения нужны GPU для обучения, T4 для проверки лимита, доступ к датасету и запуск "
            "<font name='ReportBold'>hw2_train_colab.ipynb</font>, затем "
            "<font name='ReportBold'>hw2_inference_colab.ipynb</font>. "
            "Тестовый leaderboard и 4-балльный порог нельзя считать подтвержденными без "
            "публикации правил отправки и ответа проверяющей системы."
        ),
        para(
            "Источники: <link href='https://github.com/ashaba1in/hse-nlp/blob/2026/homeworks/hw2_code_generation.md' "
            "color='#356aa5'>условие курса</link>; "
            "<link href='https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m' "
            "color='#356aa5'>описание датасета</link>.", "ReportSmall"
        ),
    ])
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
