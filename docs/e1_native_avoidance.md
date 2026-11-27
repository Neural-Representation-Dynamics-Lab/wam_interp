# Native FastWAM obstacle positive control

Before opening the pilot outcomes, we chose two development fixtures from the first-chunk experiment. We test LIBERO-10 task 0, state 0, then task 3, state 0 at its later barrier placement if the first fixture fails. Both have visible on-route and off-route solids and already passed source rendering checks. The second task previously failed a video-injection check. This pilot uses ordinary native Optional IDM inference throughout.

For each scene, use both mirrored on-route and off-route placements, one clear scene, policy seeds 2 and 12, and up to 48 controls. Replan every eight controls from the actual observation with call seed equal to the episode seed plus the call number. Keep matched cases paired. Record every native action, gripper position, robot–barrier contact at physics substeps, replan boundary, and termination. Success in this window is descriptive.

First test physical feasibility. Replaying the clear commands in the solid must hit the on-route barrier and miss the off-route barrier. At least one scripted lateral detour must reach the barrier’s center plane in the forward direction without robot–barrier contact. Try fixed normalized control biases of 0.18, 0.35 and 0.55 toward the open side over the first 20 commands. This checks that the robot can pass the fixture within the study window.

A native policy pass requires crossing that same plane without robot–barrier contact, at least 5 mm greater minimum gripper-point clearance than the matched clear path scored against the same virtual solid, and at most 5 mm off-route trajectory RMS displacement. For the RMS comparison require equal observed path lengths. Clearance and substep contact are separate metrics. Both seeds must pass on both sides of a fixture. These are prospective engineering criteria for selecting a pilot scene. A failed scripted witness leaves that comparison's physical feasibility unresolved.

A passing scene advances to a separately frozen four-condition experiment on fresh starts, using a new future at each replan. If both preselected scenes fail, retain the receipts and close this screen. A different fixture or longer readout would be a new design.

In the [obstacle study environment](e1_obstacles.md), run from the repository root:

```bash
python scripts/e1_native_avoidance.py --out results/e1_native_avoidance
```

The [design](../configs/experiments/e1_native_avoidance.json) fixes the exact geometry and reset hashes. The runner verifies each starting observation and writes a per-branch receipt and a study summary. Reuse a results folder only with the same code, checkpoint and design.
