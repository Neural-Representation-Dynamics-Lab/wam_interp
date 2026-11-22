"""Compare physical target tracking and paired task outcomes in the native relocation diagnostic."""
import json
from pathlib import Path
import numpy as np


def summarize(root):
    plan = json.loads((root / "protocol.json").read_text())
    rows, pairs = [], []
    for task in plan["plan"]:
        for layout in task["layouts"]:
            for mode in plan["conditions"]:
                matched = []
                for variant in ("official", "relocated"):
                    case = root / f"task_{task['task_id']}/state_{layout['state']}/{variant}/{mode}"
                    if not (case / "episode.json").exists():
                        continue
                    record = json.loads((case / "episode.json").read_text())
                    path = json.loads((case / "tracks.json").read_text())
                    target = record["target"]
                    t = record["first_target_contact"]
                    static_target = np.array(path[0]["objects"][target])
                    first_other = next((x for x in path if any(k != target for k in x["contacts"])), None)
                    row = dict(task=task["task_id"], state=layout["state"], condition=mode, layout=variant,
                               success=record["episode"]["success"], controls=record["episode"]["policy_steps"],
                               target_contact=t, contact_eef=path[t]["eef"] if t is not None else None,
                               first_other_contact_step=first_other["step"] if first_other else None,
                               first_other_contact_objects=first_other["contacts"] if first_other else [],
                               target_start=static_target.tolist())
                    for h in (10,32):
                        if len(path)>h:
                            row[f"eef_{h}"] = path[h]["eef"]
                            row[f"target_distance_{h}_m"] = float(np.linalg.norm(np.array(path[h]["eef"])-static_target))
                    rows.append(row)
                    matched.append(row)
                if len(matched) != 2:
                    continue
                a,b = matched
                delta = np.array(b["target_start"])[:2] - np.array(a["target_start"])[:2]
                def projection(key):
                    if a.get(key) is None or b.get(key) is None:
                        return None
                    return float((np.array(b[key])[:2]-np.array(a[key])[:2]) @ delta / (delta@delta))
                pairs.append(dict(task=a["task"],state=a["state"],condition=mode,delta_m=delta.tolist(),
                             contact_follow_fraction=projection("contact_eef"), prefix10_fraction=projection("eef_10"),
                             prefix32_fraction=projection("eef_32"), both_success=a["success"] and b["success"]))
    task_gates = {}
    for task in plan["plan"]:
        key = task["task_id"]
        available = [p for p in pairs if p["task"]==key and p["condition"]=="idm"]
        passing = [p["state"] for p in available if p["both_success"] and p["contact_follow_fraction"] is not None and p["contact_follow_fraction"] >= 0.5]
        task_gates[str(key)] = dict(available=len(available),paired_states_passing=passing,
                                  eligible=len(available)==4 and len(passing)>=3)
    comparison = []
    for row in rows:
        if row["condition"] != "idm":
            continue
        other = next((r for r in rows if r["condition"]=="first_frame" and (r["task"],r["state"],r["layout"])==(row["task"],row["state"],row["layout"])),None)
        if other:
            comparison.append(dict(task=row["task"],state=row["state"],layout=row["layout"],
                                   idm=row["success"],first_frame=other["success"]))
    return dict(complete=len(rows)==32, episodes=rows, pairs=pairs, task_gates=task_gates, same_weight_outcomes=comparison,
                gate_definition="Development criterion: at least 3/4 IDM paired states succeed on both layouts and follow at least half the physical target translation at first recorded target contact. Defined after initial state 0 bowl outcomes were becoming available; exploratory.",
                interpretation="Other contacts can include valid task fixtures. Projection uses contact at each arm's own first-contact time. Step 10/32 use synchronized time.")
if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    root=parser.parse_args().root
    out=summarize(root)
    (root/"analysis.json").write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({"episodes":len(out["episodes"]),"gates":out["task_gates"]}))
