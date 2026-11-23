#!/usr/bin/env bash
# Run independently for each task so a failed simulator check has an explicit denominator.
set -euo pipefail
screen=${1:?Pass a directory with expert screening and smoke receipts}
task=${2:?Pass a screened task name}
py=${PYTHON:-python}
root="$screen/$task"
lock=${E1_GPU_LOCK:-"$screen/../research_gpu.lock"}
gpu() { flock "$lock" "$@"; }
# A fresh simulator process can reclaim Vulkan camera buffers. Retry only this
# specific allocation failure and only while new scene/seed receipts are being sealed.
run_stage() {
 local stage=$1 split=$2 count after attempt=1 log="$root/logs/$1.log"
 local marker=longwam_${stage}.json
 case "$stage" in calibrate) marker=longwam_behavior.json;; sources) marker=longwam_source.json;; esac
 while true; do
  count=$(find "$root/$split" -name "$marker" 2>/dev/null | wc -l) || count=0
  if gpu "$py" scripts/e1_confirm_robotwin.py "$stage" "${args[@]}" --manifest "$root/$split.json" --out "$root/$split" >"$log" 2>&1; then return 0; fi
  if ! grep -Fq "RuntimeError: cannot create buffer" "$log"; then return 1; fi
  after=$(find "$root/$split" -name "$marker" | wc -l)
  if (( after <= count )); then return 1; fi
  mv "$log" "$root/logs/${stage}_allocation_${attempt}_$(date +%s).log"
  attempt=$((attempt + 1))
 done
}
mkdir -p "$root/logs"
"$py" scripts/e1_robotwin_manifest.py --screen-dir "$screen" --out "$root" --tasks "$task"
args=(--root "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT}" --weights "${ROBOTWIN_IDM:?Set ROBOTWIN_IDM}" --screen-dir "$screen")
(
 flock 9
 if [[ ! -s "$screen/smoke_$task/fidelity.json" ]]; then
  "$py" scripts/e1_robotwin_clip.py --root "$ROBOTWIN_ROOT" --weights "$ROBOTWIN_IDM" --task "$task" --screen "$screen/dev_$task.json" --out "$screen/smoke_$task" >"$root/logs/native_retry.log" 2>&1
 fi
) 9>"$lock"
test -s "$screen/smoke_$task/fidelity.json"
if ! "$py" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$screen/smoke_$task/fidelity.json"; then
  echo "Development unedited-video fidelity has not passed" >"$root/stop_reason.txt"
  exit 0
fi
run_stage calibrate dev
"$py" - "$root" <<'PY'
import sys,json
from pathlib import Path
root=Path(sys.argv[1])
spec=json.loads((root/"dev.json").read_text())
task=spec["tasks"][0]
usable=0
for idx in spec["states"][task]:
    files=[root/f"dev/task_{task}/state_{idx}/seed_{s}/longwam_behavior.json" for s in spec["seeds"]]
    usable+=all(f.exists() and json.loads(f.read_text())["eligible"] for f in files)
(root/"development_gate.json").write_text(json.dumps(dict(usable=usable,attempted=len(spec["states"][task]),passed=usable>=2)))
if usable<2:
    (root/"stop_reason.txt").write_text(f"Development source and sham eligible on {usable} of {len(spec['states'][task])} independent starts. At least two are required to fit a direction template.")
PY
if ! "$py" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$root/development_gate.json"; then exit 0; fi
for stage in sources features; do
 run_stage "$stage" held
done
"$py" scripts/e1_confirm_stimuli.py --manifest "$root/held.json" --out "$root/held"
"$py" scripts/e1_confirm_analysis.py gate --root "$root" --backend longwam
run_stage behavior held
"$py" scripts/e1_confirm_analysis.py report --root "$root" --backend longwam
"$py" scripts/e1_confirm_figures.py --root "$root" --out "$root/figures"
