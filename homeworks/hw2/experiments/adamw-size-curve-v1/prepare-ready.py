from pathlib import Path
import json
from homeworks.hw2.checkpoint_io import sha256
from homeworks.hw2.adamw_size_curve import PLAN

exp=Path("homeworks/hw2/experiments/adamw-size-curve-v1")
average=Path("homeworks/hw2/experiments/full-pass-average-cpu-v1")
selection=json.loads((average/"selection.json").read_text())
assert "selected" in selection and len(selection["pilot"])==3
assert selection["training_updates"]==0 and not selection["public_test_used_for_selection"]
# A winning average must be exported/scored before allocating a different GPU run.
assert selection["selected"] is None, "Finish the confirmed averaging candidate first"
proof=json.loads((average/"final-drive-verification.json").read_text())
assert proof["download_verified"]
source=json.loads((exp/"preparation-source.json").read_text())
assert all(x["byte_verified"] for x in source["files"])
ready=json.loads(Path("homeworks/hw2/experiments/matched-curve-v1/ready.json").read_text())
reference=json.loads((exp/"reference.json").read_text())
cpu=json.loads((exp/"cpu-preflight.json").read_text())
assert cpu["cpu_verified"] and cpu["fresh_index_sha256"]==reference["index_manifest_sha256"]
ready.update(source_commit=source["source_commit"],
 source_sha256=next(x["sha256"] for x in source["files"] if x["name"]=="source.tar"),
 launcher_sha256=sha256(exp/"launch.py"),plan=PLAN,cpu_preflight=cpu,
 runtime_versions=reference["runtime_versions"],reference_sha256=sha256(exp/"reference.json"),
 prerequisite_cpu_averaging_completed=True,
 cpu_averaging_selection_sha256=sha256(average/"selection.json"),
 native_muon_runtime_required=False,
 tests=dict(cpu_only=True,driver_unit_tests_passed=7))
(exp/"ready.json").write_text(json.dumps(ready,indent=2)+"\n")
print("READY_MANIFEST_CREATED")
