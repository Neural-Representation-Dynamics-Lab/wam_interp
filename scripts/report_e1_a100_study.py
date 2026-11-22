"""Compile portable, machine-readable results from completed A100 experiment records."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

from analyze_native_replan import summarize

def load(p): return json.loads(p.read_text())
def mean(x): return float(np.mean(x)) if x else None

def episode_cost(records, peak_field):
    median_times = []
    peaks = []
    for record in records:
        elapsed = [call["inference_latency_ms"] for call in record["episode"]["inference_metadata"]]
        if elapsed:
            median_times.append(float(np.median(elapsed)))
        peaks.append(record[peak_field])
    return {"episode_mean_of_median_call_ms": mean(median_times),
            "max_allocated_gib": max(peaks) / 2**30 if peaks else None}


def stress_test(directory):
    cases = [load(p) for p in sorted(directory.glob("state_*/*/*/episode.json"))]
    results = {}
    for variant in ("original", "swapped"):
        results[variant] = {}
        for condition in ("idm", "first_frame"):
            cell = [r for r in cases if r["variant"] == variant and r["condition"] == condition]
            results[variant][condition] = {
                "episodes": len(cell),
                "success": sum(r["episode"]["success"] for r in cell),
                "first_contacts": [r["first_contact"] for r in cell],
                "soup_contact_episodes": sum(r["first_target_contact"] is not None for r in cell),
                **episode_cost(cell, "peak_allocated_bytes"),
            }
    return {"complete": len(cases) == 16, "completed_episodes": len(cases), "cells": results}


def contrast_statistics(groups):
    stats = {}
    for metric in ("relevant_half_contrast_m", "nuisance_half_contrast_m", "selective_half_contrast_m"):
        state_values = [mean([r[metric] for r in rows]) for rows in groups.values()]
        stats[metric] = mean(state_values)
        if len(state_values) > 1:
            rng = np.random.default_rng(20261122)
            bootstrap = rng.choice(state_values, size=(10000, len(state_values)), replace=True).mean(axis=1)
            stats[metric + "_state_bootstrap_95"] = np.quantile(bootstrap, [.025, .975]).tolist()
    return stats


def build(results):
    root=results/"e1_native_replan"
    p0=summarize(root)
    summary={"hardware":load(root/"protocol.json")["gpu"],"native_complete":p0["complete"],
               "native_episodes":len(p0["episodes"]),"native_target_control":p0["task_gates"]}
    native_records = [load(p) for p in sorted(root.glob("task_*/state_*/*/*/episode.json"))]
    summary["native_cost"] = {
        condition: episode_cost([r for r in native_records if r["condition"] == condition], "peak_gpu_memory_bytes")
        for condition in ("idm", "first_frame")
    }
    summary["native_cost_scope"] = "Synchronized instrumented wall time. IDM includes forecast copying and saving."
    summary["large_swap_replan"] = stress_test(results / "e1_large_swap_replan")
    native={}
    for r in p0["episodes"]:
        key=f"task_{r['task']}/{r['layout']}/{r['condition']}"
        cell=native.setdefault(key,dict(episodes=0,successes=0,first_contact_steps=[]))
        cell["episodes"]+=1
        cell["successes"]+=int(r["success"])
        if r["target_contact"] is not None:cell["first_contact_steps"].append(r["target_contact"])
    summary["native_cells"]=native
    summary["paired_success_outcomes"]=dict(Counter(f"idm_{int(p['idm'])}_first_frame_{int(p['first_frame'])}" for p in p0["same_weight_outcomes"]))
    summary["paired_task_success_unit"]="same task, initial state and layout; two layouts per state"
    landmarks=defaultdict(list)
    for p in sorted(root.glob("task_*/state_*/*/idm/alignment.json")):
        rec=load(p)
        for key in ("eef",rec["target"]):
            values=[v for r in rec["rows"] for v in r.get(key,[])]
            usable=[v for v in values if v is not None]
            cv=[v for v in usable if v["constant_velocity_error_px"] is not None]
            moving=[v for v in usable if v["stationary_error_px"]>=2]
            endpoints=[r[key][-1] for r in rec["rows"] if r.get(key) and r[key][-1] is not None]
            labels=("gripper" if key=="eef" else "target")
            landmarks[labels].append(dict(episode=str(p.parent.relative_to(root)),decisions=len(rec["rows"]),
                           valid=len(usable),eligible=len(values),valid_endpoints=len(endpoints),
                           mean_error=mean([v["error_px"] for v in usable]),
                           endpoint_error=mean([v["error_px"] for v in endpoints]),
                           stationary=mean([v["stationary_error_px"] for v in usable]),
                           constant_velocity=mean([v["constant_velocity_error_px"] for v in cv]),
                           forecast_on_cv_subset=mean([v["error_px"] for v in cv]),
                           moving=len(moving),forecast_on_moving=mean([v["error_px"] for v in moving]),
                           stationary_on_moving=mean([v["stationary_error_px"] for v in moving])))
    summary["alignment"]={}
    for key,episodes in landmarks.items():
        stats={k:sum(e[k] for e in episodes) for k in ("decisions","valid","eligible","moving","valid_endpoints")}
        for k in ("mean_error","endpoint_error","stationary","constant_velocity","forecast_on_cv_subset","forecast_on_moving","stationary_on_moving"):
            stats[k]=mean([e[k] for e in episodes if e[k] is not None])
        summary["alignment"][key]=dict(episodes=len(episodes),**stats,per_episode=episodes)
    summary["alignment_scope"]="External-view 2D displacement near projected landmarks. Each replan scored at executed offsets 4 and 8 if available. Means weight episodes equally; untracked points excluded with denominators. Human verification pending."
    model_dir=results/"e1_model_audits"
    summary["architectures"]={}
    for name in ("fasterwam","lingbot_va","longwam_idm","openwam"):
        d=model_dir/name
        row={}
        for p in sorted(d.glob("libero_libero_10_*.json")):
            r=load(p)
            row.update(completed=r.get("completed"), error=r.get("error"),results=[{"success":e["success"],"steps":e["policy_steps"]} for t in r["tasks"] for e in t["trials"]])
        if (d/"parity.json").is_file():
            row["parity"]= [{k:ch[k] for k in ("call","layer","stage","occurrence","capture_max_abs","identity_max_abs","same_features","ablation_max_abs","peak_allocated_bytes") if k in ch} for ch in load(d/"parity.json")["chunks"]]
        if (d/"scene_features/results.json").is_file():
            scenes=load(d/"scene_features/results.json")["rows"]
            row["scene_cases"]=dict(Counter(a["status"] for a in scenes))
            row["scene_first10_selective_mean_m"]=mean([a["selective_difference_m"] for a in scenes if a["status"]=="sampled"])
        summary["architectures"][name]=row
    for folder,key in (("e1_sliding_targets","physical_slide"),("e1_prescribed_slide","prescribed_slide")):
        slide=results/folder
        entry=load(slide/"status.json") if (slide/"status.json").exists() else {}
        for stage in ("development","held_out"):
            paths=sorted(slide.glob(f"{stage}/*/*/*/case.json"))
            trials=[load(p) for p in paths]
            groups=defaultdict(list)
            for p,row in zip(paths,trials):
                if "selective_half_contrast_m" in row:
                    groups[(p.parents[2].name,row["state"])].append(row)
            if stage=="held_out":
                groups={k:rs for k,rs in groups.items() if {r["seed"] for r in rs}=={0,1}}
            stats=contrast_statistics(groups)
            per_task = {}
            for task in sorted({p.parents[2].name for p in paths}):
                task_groups = {k: rs for k, rs in groups.items() if k[0] == task}
                task_rows = [r for p, r in zip(paths, trials) if p.parents[2].name == task]
                per_task[task] = {"attempted":len(task_rows), "eligible_states":len(task_groups),
                                  "source_passed":sum(r["source_pass"] for r in task_rows),
                                  "sham_passed":sum(bool(r.get("sham_pass")) for r in task_rows),
                                  **contrast_statistics(task_groups)}
            entry[stage]=dict(total=len(trials),passed_source=sum(bool(r["source_pass"]) for r in trials),
                       passed_sham=sum(bool(r.get("sham_pass")) for r in trials),eligible_states=len(groups),
                       alternate_native_contacts=sum(bool(r.get("alternate_instruction",{}).get("native_first_contact",{}).get("valid")) for r in trials),
                       alternate_interactions=[r["alternate_instruction"]["instruction_by_property_interaction_m"] for r in trials if "instruction_by_property_interaction_m" in r.get("alternate_instruction",{})],
                       stats=stats, per_task=per_task)
            interval=stats.get("nuisance_half_contrast_m_state_bootstrap_95")
            if interval is not None and stage=="held_out":
                entry[stage]["nuisance_interval_within_5mm"]=bool(interval[0]>=-.005 and interval[1]<=.005)
        summary[key]=entry
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    data=build(args.results)
    args.out.write_text(json.dumps(data,indent=2,allow_nan=False))
    print("Compiled",data["native_episodes"],"native episodes")
