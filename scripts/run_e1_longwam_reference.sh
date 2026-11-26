#!/usr/bin/env bash
# Run from the study checkout against a completed shared-video study.
set -euo pipefail
shared=${1:?Pass the original shared-video study directory}
out=${2:?Pass a separate output directory}
py=${PYTHON:-python}
lock=${E1_GPU_LOCK:-"$shared/research_gpu.lock"}
mkdir -p "$out/logs"
gpu() { flock "$lock" "$@"; }
"$py" scripts/e1_longwam_reference.py register --shared "$shared" --out "$out"
for split in dev held; do
  "$py" scripts/e1_longwam_reference.py references --shared "$shared" --out "$out" --split "$split"
  if ! "$py" -c 'import sys,json; sys.exit(not all(r["copied"] for r in json.load(open(sys.argv[1]))))' "$out/${split}_reference_ledger.json"; then
    gpu "$py" scripts/e1_confirm.py references --backend longwam --manifest "$out/$split.json" --out "$out/$split" >"$out/logs/references_$split.log" 2>&1
  fi
done
"$py" scripts/e1_confirm_sources.py --backend longwam --workers 3 --manifest "$out/dev.json" --out "$out/dev" >"$out/logs/dev_sources.log" 2>&1
flock "$out/held_sources.lock" "$py" scripts/e1_confirm_sources.py --backend longwam --workers 3 --manifest "$out/held.json" --out "$out/held" >"$out/logs/held_sources.log" 2>&1 &
sources=$!
trap 'kill "$sources" 2>/dev/null || true' EXIT
gpu "$py" scripts/e1_confirm.py calibrate --backend longwam --manifest "$out/dev.json" --out "$out/dev" >"$out/logs/calibrate.log" 2>&1
wait "$sources"
trap - EXIT
"$py" scripts/e1_confirm_stimuli.py --manifest "$out/held.json" --out "$out/held"
gpu "$py" scripts/e1_confirm.py features --backend longwam --manifest "$out/held.json" --out "$out/held" >"$out/logs/features.log" 2>&1
"$py" scripts/e1_confirm_analysis.py gate --root "$out" --backend longwam
readouts=()
trap 'if (( ${#readouts[@]} )); then kill "${readouts[@]}" 2>/dev/null || true; fi' EXIT
for task in 5 3 7; do
  "$py" scripts/e1_confirm_execute.py --task "$task" --wait --backend longwam --manifest "$out/held.json" --out "$out/held" >"$out/logs/execute_$task.log" 2>&1 &
  readouts+=("$!")
done
gpu "$py" scripts/e1_confirm.py commands --backend longwam --manifest "$out/held.json" --out "$out/held" >"$out/logs/commands.log" 2>&1
for pid in "${readouts[@]}"; do wait "$pid"; done
readouts=()
"$py" scripts/e1_confirm_analysis.py report --root "$out" --backend longwam
"$py" scripts/e1_longwam_reference.py compare --shared "$shared" --out "$out"
"$py" scripts/e1_confirm_figures.py --root "$out" --out "$out/figures"
