#!/usr/bin/env bash
# Reproduce the FastWAM Optional IDM obstacle smoke or held-out LIBERO study.
set -euo pipefail

if [[ $# != 2 || ( $1 != smoke && $1 != heldout ) ]]; then
  echo "Usage: bash scripts/run_e1_obstacle_fastwam.sh {smoke|heldout} OUTPUT_DIR" >&2
  exit 2
fi

mode=$1
out=$2
configs=configs/experiments
held=$configs/e1_obstacle_fastwam_task0.json
py=${PYTHON:-python}

run_cases() {
  local manifest=$1 folder=$2
  local args=(--backend fastwam --manifest "$manifest")
  "$py" scripts/e1_confirm.py references "${args[@]}" --out "$folder/refs"
  "$py" scripts/e1_obstacle.py sources "${args[@]}" --refs "$folder/refs" --out "$folder/base"
  "$py" scripts/e1_obstacle_offroute.py "${args[@]}" --source "$folder/base" --out "$folder/qualified"
  "$py" scripts/e1_obstacle_pilot.py "${args[@]}" --out "$folder/qualified"
}

if [[ $mode == smoke ]]; then
  run_cases "$configs/e1_obstacle_smoke.json" "$out/smoke"
  "$py" - "$out/smoke/qualified/fastwam/task_0/state_0/seed_0/behavior.json" <<'PY'
import json
import sys
from pathlib import Path

case = Path(sys.argv[1])
if not case.is_file() or not json.loads(case.read_text())["passed"]:
    sys.exit(f"Smoke failed. Check sources.json and calibration.json in {case.parent}")
print(f"Paired behavior and native replay passed: {case}")
PY
  exit 0
fi

sha256sum -c <<EOF
5a0ae85dd7dcdcca19d60e003fe2e3b076789e2f340b5a8cf5383dced7775f30  $held
f08af7a12b840cafc010133ea4fe5245e3723777adae6f0174c774a1cbc00238  $configs/e1_obstacle_unsigned_supplement.json
EOF
run_cases "$configs/e1_obstacle_dev.json" "$out/development"
run_cases "$held" "$out/held"
"$py" scripts/e1_obstacle_held_report.py --manifest "$held" \
  --development "$out/development/qualified" --held "$out/held/qualified" --out "$out/heldout.json"
