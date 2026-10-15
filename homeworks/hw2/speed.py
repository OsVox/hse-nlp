"""Небольшие оптимизации из лекции 3: точность и группировка по длине."""

from itertools import islice

import torch


def precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    if device.type != "cuda" and dtype != torch.float32:
        raise ValueError("fp16/bf16 в этом скрипте требуют CUDA; используйте fp32")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU не поддерживает bf16; используйте fp16 или fp32")
    return dtype


def group_batches(batches, tokenizer, batch_size: int, window: int):
    """Сортируем небольшое окно, сохраняя все примеры и границы выборки."""
    if window < 1:
        raise ValueError("group window must be positive")
    if window == 1:
        yield from batches
        return
    batches = iter(batches)
    while chunk := list(islice(batches, window)):
        rows = [row for batch in chunk for row in batch]
        source = tokenizer([row["question"] for row in rows], truncation=True,
                           max_length=128)["input_ids"]
        target = tokenizer([row["code"] for row in rows], truncation=True,
                           max_length=192)["input_ids"]
        # Учитываем и вопрос, и код: паддинг есть у энкодера и декодера.
        order = sorted(range(len(rows)), key=lambda i: (len(source[i]), len(target[i])))
        for start in range(0, len(order), batch_size):
            yield [rows[i] for i in order[start:start + batch_size]]
