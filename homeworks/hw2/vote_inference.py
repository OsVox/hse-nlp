"""Inference-only execution voting; every saved program is a model decode."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from .candidate_vote import select_indices
from .checkpoint_io import package_jsonl, sha256
from .numeric_eval import execute_numeric


def run(model_dir, cache_dir, output_dir, *, beams=4, batch_size=64):
    import pyarrow.parquet as pq
    import torch
    from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
    from .data import fetch_parquet
    assert beams == 4 and batch_size > 0, 'Use the tested four-candidate policy'
    assert not output_dir.exists(), 'Preserve previous exports'
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = T5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True).to(device).eval()
    test = fetch_parquet('data/test/test.parquet', cache_dir)
    assert sha256(test) == '8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a'
    questions = pq.read_table(test, columns=['question'])['question'].to_pylist()
    question_hash = hashlib.sha256(json.dumps(questions, ensure_ascii=False).encode()).hexdigest()
    assert len(questions) == 1919
    assert question_hash == '214746f8d4bc918b6fa1fde8ca3185a518a60637f239a2c45efa9a8143f612d3'
    if device == 'cuda':
        torch.cuda.synchronize()
    began = time.perf_counter()
    encoded = tokenizer(questions, truncation=True, max_length=128,
                        return_token_type_ids=False)
    order = sorted(range(len(questions)), key=lambda i: len(encoded['input_ids'][i]))
    groups, ids = [None] * len(questions), [None] * len(questions)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = tokenizer.pad([{k: v[i] for k, v in encoded.items()} for i in indices],
                                  padding=True, return_tensors='pt').to(device)
            generated = model.generate(**batch, max_new_tokens=192, num_beams=beams,
                                       num_return_sequences=beams, do_sample=False,
                                       use_cache=True)
            tokens = generated.tolist()
            codes = tokenizer.batch_decode(tokens, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            for j, index in enumerate(indices):
                groups[index] = codes[j*beams:(j+1)*beams]
                ids[index] = tokens[j*beams:(j+1)*beams]
    if device == 'cuda':
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - began
    unique = list(dict.fromkeys(code for group in groups for code in group))
    with ThreadPoolExecutor(max_workers=4) as pool:
        execution = dict(zip(unique, pool.map(execute_numeric, unique)))
    indices = [select_indices([execution[c] for c in group])['strict_majority'] for group in groups]
    programs = [group[i] for group, i in zip(groups, indices)]
    selected = [group[i] for group, i in zip(ids, indices)]
    assert all(programs)
    assert programs == tokenizer.batch_decode(selected, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir/'predictions.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['code'])
        writer.writeheader(); writer.writerows({'code': c} for c in programs)
    seconds = time.perf_counter() - began
    bundled = Path('source_commit.txt')
    commit = (bundled.read_text().strip() if bundled.exists() else
              subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    provenance = dict(source_commit=commit, model_sha256=sha256(model_dir/'model.safetensors'),
        output_sha256=sha256(csv_path), questions_sha256=question_hash,
        generation_seconds=seconds, decode_seconds=decode_seconds,
        gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
        precision='torch.float32', batch_size=batch_size, num_beams=beams,
        num_return_sequences=beams, max_new_tokens=192, handwritten_repairs=0,
        selection='strict_majority_execution', reference_used_for_selection=False,
        selected_token_ids=selected, selected_candidate_indices=indices,
        changed_from_top=sum(i != 0 for i in indices))
    (output_dir/'predictions.provenance.json').write_text(json.dumps(provenance))
    with (output_dir/'candidates.jsonl').open('w', encoding='utf-8') as f:
        for i, group in enumerate(groups):
            f.write(json.dumps(dict(row=i, codes=group, candidate_token_ids=ids[i],
                execution=[execution[c] for c in group], selected=indices[i]), ensure_ascii=False)+'\n')
    digest = package_jsonl(csv_path, output_dir/'predictions.jsonl')
    for name in ['checkpoint.complete.json', 'config.json', 'generation_config.json',
                 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                 'step.json', 'run_config.json', 'validation.json']:
        if (model_dir/name).exists():
            shutil.copyfile(model_dir/name, output_dir/name)
            assert sha256(model_dir/name) == sha256(output_dir/name)
    result = dict(step=json.loads((model_dir/'step.json').read_text())['step'],
        source_commit=commit, model_sha256=provenance['model_sha256'], rows=len(programs),
        jsonl_sha256=digest, generation_seconds=seconds, decode_seconds=decode_seconds,
        device=device, num_beams=beams, batch_size=batch_size,
        selection=provenance['selection'], changed_from_top=provenance['changed_from_top'],
        bot_status='pending', public_pass_at_1=None, t4_timing_verified=False,
        question_order='official data/test/test.parquet', handwritten_repairs=0)
    (output_dir/'results.complete.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(**vars(parser.parse_args()))
