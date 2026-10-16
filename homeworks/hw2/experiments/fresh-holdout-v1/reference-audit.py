from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json,hashlib
from homeworks.hw2.numeric_eval import load_examples,execute_numeric
p=Path('/Users/andrejvorsin/Downloads/hw2-fresh-holdout-v1-restored/questions.jsonl')
rows=load_examples(None,p)
codes=sorted({r['code'] for r in rows})
with ThreadPoolExecutor(max_workers=4) as pool:result=dict(zip(codes,pool.map(execute_numeric,codes)))
counts={}
for name,start,end in [('pilot',0,1024),('confirmation',1024,2048)]:
 statuses=Counter(result[r['code']]['status'] for r in rows[start:end]);counts[name]=dict(rows=end-start,usable_references=statuses['ok'],status_counts=dict(statuses))
out=Path('homeworks/hw2/experiments/fresh-holdout-v1/reference-audit.json')
out.write_text(json.dumps(dict(questions_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),subsets=counts,selection_changed=False,targets_rewritten=False,model_evaluated=False,note='Restricted numeric runner diagnostics only; unsupported references are not automatically incorrect. No rows are removed based on these results.'),indent=2)+'\n')
print(json.dumps(counts),flush=True)
