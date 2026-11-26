"""Select the first mirrored off-route distance passing physical and image checks, blind to actions."""

import argparse
import json
from pathlib import Path

import numpy as np
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import save
from wam_interp.experiments.confirm_replay import anchor_hash
from wam_interp.experiments.obstacle import placement
from wam_interp.experiments.obstacle_libero import hold_after, rollout, scene
from wam_interp.experiments.simulator_video import paired_camera_clip


def refine(source, dest, task, state):
    meta = json.loads((source / "sources.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "sources.json").exists():
        return
    if not all(meta["placements"].get(k, {}).get("passed") for k in ("minus", "plus")):
        meta.update(passed=False, reason="on-route source gate failed")
        save(dest / "sources.json", meta)
        return
    with np.load(source / "sources.npz") as z:
        data = {k: z[k] for k in z.files}
    meta["off_route_attempts"] = []
    prefix = meta["depicted_native_prefix_steps"]
    for distance, forward_offset in (
        (0.024, 0.09),
        (0.024, -0.09),
        (0.024, 0.13),
        (0.024, -0.13),
        (0.095, 0),
        (0.115, 0),
        (0.08, 0),
        (0.14, 0),
        (0.165, 0),
    ):
        rows = {}
        for name, side in (("minus", -1), ("plus", 1)):
            key = "off_" + name
            box = placement(
                data["native_path"],
                side,
                off_route=True,
                off_distance=distance,
                forward_offset=forward_offset,
                station=meta.get("placement_station", 28),
            )
            with scene(
                None, task, state, meta["state"], meta["task"], meta["seed"], box
            ) as (env, kw, obs, _):
                anchor = anchor_hash(obs)
                path, qpos, cams, contacts = rollout(
                    env, kw, obs, data["commands"], frames=True, barrier=True
                )
            if len(path) != 33:
                rows[key] = {"passed": False, "reason": "short reference"}
                continue
            path, qpos = hold_after(path, prefix), hold_after(qpos, prefix)
            clip = hold_after(
                paired_camera_clip(cams["agentview"], cams["wrist"]), prefix // 4
            )
            changed = (clip != data["clear"]).any(-1).sum((1, 2))
            area_ratio = float(
                changed[1:].mean()
                / np.mean(meta["placements"][name]["changed_pixels"][1:])
            )
            physics = float(np.max(np.abs(qpos - data["clear_qpos"])))
            motion = float(np.max(np.abs(path - data["clear_path"])))
            rows[key] = {
                "geometry": box.as_dict(),
                "anchor": anchor,
                "all_contacts": contacts,
                "max_qpos_delta": physics,
                "max_eef_delta_m": motion,
                "changed_pixels": changed.tolist(),
                "mean_changed_area_ratio": area_ratio,
                "passed": bool(
                    not contacts
                    and physics <= 1e-8
                    and motion <= 1e-8
                    and changed.min() >= 100
                    and 0.25 <= area_ratio <= 4
                ),
            }
            data.update({key: clip, key + "_path": path, key + "_qpos": qpos})
        meta["off_route_attempts"].append(
            {"lateral_m": distance, "forward_m": forward_offset, "placements": rows}
        )
        meta["placements"].update(rows)
        if all(row["passed"] for row in rows.values()):
            meta.update(
                passed=True,
                off_route_lateral_m=distance,
                off_route_forward_m=forward_offset,
            )
            break
        meta["passed"] = False
    if meta["passed"]:
        np.savez_compressed(dest / "sources.npz", **data)
    save(dest / "sources.json", meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    suite = _get_suite("libero_10")
    for t in m["tasks"]:
        task, states = suite.get_task(t), initial_states_for_task(suite, t)
        for i in m["states"][str(t)]:
            for s in m["seeds"]:
                rel = Path(a.backend) / f"task_{t}/state_{i}/seed_{s}"
                refine(a.source / rel, a.out / rel, task, states[i])
                print(rel, flush=True)
