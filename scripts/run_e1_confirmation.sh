#!/usr/bin/env bash
# Begin after development calibration. Run from the repository in the LIBERO environment.
set -euo pipefail
root=${1:?Pass the experiment directory containing dev.json and dev/}
py=${PYTHON:-python}
design=${E1_DESIGN:-configs/experiments/e1_confirmation.json}
mkdir -p "$root/logs"
"$py" scripts/e1_confirm_analysis.py freeze --root "$root" --design "$design"
manifest="$root/held.json"
out="$root/held"
mkdir -p "$out"
sha256sum "$manifest" >"$root/held.sha256"
run_model() {
  "$py" scripts/e1_confirm.py "$1" --backend "$2" --manifest "$manifest" --out "$out"
}
run_model references fastwam >"$root/logs/references.log" 2>&1
"$py" scripts/e1_confirm_sources.py --manifest "$manifest" --out "$out" --workers "${E1_RENDER_WORKERS:-4}" >"$root/logs/sources.log" 2>&1
for model in fastwam longwam; do
  run_model features "$model" >"$root/logs/features_$model.log" 2>&1
  "$py" scripts/e1_confirm_analysis.py gate --backend "$model" --root "$root"
done
mapfile -t tasks < <("$py" -c 'import json,sys;print(*json.load(open(sys.argv[1]))["tasks"],sep="\n")' "$manifest")
readouts=()
finish_children() {
  if (( ${#readouts[@]} )); then kill "${readouts[@]}" 2>/dev/null || true; fi
}
trap finish_children EXIT
for model in fastwam longwam; do
  for task in "${tasks[@]}"; do
    "$py" scripts/e1_confirm_execute.py --backend "$model" --manifest "$manifest" --out "$out" --wait --task "$task" >"$root/logs/execute_${model}_${task}.log" 2>&1 &
    readouts+=("$!")
  done
  run_model commands "$model" >"$root/logs/commands_$model.log" 2>&1
done
for pid in "${readouts[@]}"; do wait "$pid"; done
readouts=()
"$py" scripts/e1_confirm_analysis.py report --root "$root"
"$py" scripts/e1_confirm_figures.py --root "$root" --out "$root/figures"
