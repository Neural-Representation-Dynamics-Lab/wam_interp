# Native coherent relocation positive control

Purpose: establish whether FastWAM Optional IDM follows a named object's new physical position before interpreting a fixed-motion future-video intervention. This test changes the current physical scene; it is not a future-transplant experiment or a benchmark success evaluation.

Four explicit arrangements (initial-state keys 34-37, with frozen small offsets) are each tested in original and physically swapped conditions. Soup and tomato exchange their free-joint world XY coordinates before 30 ordinary warm-up controls. Each object retains its own Z, quaternion, velocity, mass, mesh, UVs, materials and collision shape. No appearance-only rendering patch is active. MuJoCo forward kinematics recomputes all attached visual and collision geometry together. This follows the [MuJoCo free-joint and position computation semantics](https://mujoco.readthedocs.io/en/3.3.2/computation/index.html).

Before inference, audit all geometry translations, unchanged model parameters, native camera timestamps, reset replay equality, visibility, settling and overlap. Each target must occupy at least 16 native agent-camera pixels; wrist visibility is reported without requiring all targets in its narrow view. Paired robot proprioception and control values must agree within absolute 1e-12; exactness and differences are also reported. Unmoved objects must agree within 1e-8 m, swapped XY slots within 1 mm, and warm-up XY settling within 5 mm. Any failure stops the entire source batch. Actually inspect all four image pairs; independent human review remains pending unless supplied.

Version 1 stopped before any policy inference: a bitwise paired-start criterion detected maximum control roundoff of 1.03e-15 and EEF-position roundoff of 1.74e-18 m. All relocation, overlap, settling and visibility gates passed. Version 2 introduces the numerical 1e-12 start tolerance, retaining exactness diagnostics, all arrangements, behavioural criteria and repeat/replay equality checks. The failed version is preserved. This engineering revision occurred before policy outcomes.

Run all 4 layouts x 2 physical arrangements x 3 instructions x 2 seeds = 48 native policy branches. Run 12 additional original-scene seed-zero repeats to check determinism. Each branch executes one 32-control chunk, with 10 inference steps and bf16 precision, without replanning. The source checkpoint, statistics, base assets, package versions, input hashes and full model configuration are recorded. Every source case is checked again against its frozen camera/proprio/physics arrays before inference. No outcome-dependent case selection.

For each layout and seed, project endpoints onto the original soup-to-tomato XY axis. A passing case requires original tomato-minus-soup separation >=80 mm, swapped separation <=-80 mm, soup relocation shift >=+80 mm, tomato shift <=-80 mm and untouched cream-cheese 3D shift <=20 mm. Require zero robot-scene contacts and <=1 mm object movement during this approach window. Report all criteria separately: failure of the untouched-target gate is distinct from failure to follow moved objects. A layout must pass both seeds; at least 3/4 layouts must pass for this bounded positive control to be considered validated. Do not claim equivalence or generalization from a failure/pass alone.

Continuous descriptive outcome: (soup relocation shift - tomato relocation shift)/2. Average seeds within layouts and bootstrap whole layouts with the existing fixed seed; with four designed arrangements on one table, intervals are descriptive only. Report distances to each instruction's actual target, contacts, all failures, synchronized inference latency and peak GPU allocation. The original soup BDDL success signal is retained but is not a success measure for tomato/cream instructions. Changing current observations changes the model's native future too; this cannot isolate direct visual perception from future use.

After all outcomes, independently replay every physical arrangement/instruction/layout at seed zero (24 replays), comparing all saved camera, robot and object arrays and integration-step contact records exactly. Preserve failures; no threshold changes after outcomes.

Linux setup uses the existing E1 environment/checkpoint and LIBERO assets. From the repository root:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
OUT=results/e1_physical_relocation_20261121_v2
.venv-e1/bin/python scripts/e1_physical_relocation.py prepare --output "$OUT"
.venv-e1/bin/python scripts/e1_physical_relocation.py sources --output "$OUT"
# Inspect all state_*/source.jpg; record actual review in source_review.json.
.venv-e1/bin/python scripts/e1_physical_relocation.py run --output "$OUT"
.venv-e1/bin/python scripts/e1_physical_relocation.py analyze --output "$OUT"
.venv-e1/bin/python scripts/e1_physical_relocation.py replay --output "$OUT"
```

Each stage checks the frozen protocol/source and refuses silent restart. This experiment does not automatically launch future-video tests. Physical target following would be a prerequisite for designing that next intervention, not evidence of future semantic faithfulness itself.
