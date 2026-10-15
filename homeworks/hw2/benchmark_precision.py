"""Сравнение точности вычислений на одном checkpoint и тестовом наборе.

Запуск из корня репозитория на GPU:
python -m homeworks.hw2.benchmark_precision --precision fp16
"""
import argparse
import csv
import json
from pathlib import Path

import torch

from .inference import generate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', choices=['fp16', 'bf16'], default='fp16')
    parser.add_argument('--model-dir', type=Path, default=Path('homeworks/hw2/model'))
    parser.add_argument('--cache-dir', type=Path, default=Path('homeworks/hw2/cache'))
    parser.add_argument('--output-dir', type=Path, default=Path('homeworks/hw2/precision_benchmark'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-new-tokens', type=int, default=192)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('Этот замер требует CUDA; CPU не подтверждает скорость на T4')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    outputs = {}
    for precision in ['fp32', args.precision]:
        destination = args.output_dir / f'{precision}.csv'
        count, seconds = generate_predictions(
            args.model_dir, destination, args.cache_dir, precision=precision,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        )
        results[precision] = {'rows': count, 'seconds': seconds}
        with destination.open(newline='', encoding='utf-8') as f:
            outputs[precision] = [row['code'] for row in csv.DictReader(f)]
    if len(outputs['fp32']) != len(outputs[args.precision]):
        raise RuntimeError('Число ответов в двух запусках отличается')
    record = {
        'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'model_dir': str(args.model_dir),
        'batch_size': args.batch_size,
        'max_new_tokens': args.max_new_tokens,
        'runs': results,
        'changed_answers': sum(a != b for a, b in zip(outputs['fp32'], outputs[args.precision])),
        'note': 'Один запуск на вариант, fp32 первым. Это предварительный замер, не pass@1. '
                'Оба файла сохранены отдельно; отправляемые ответы не заменены.',
    }
    (args.output_dir / 'results.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
