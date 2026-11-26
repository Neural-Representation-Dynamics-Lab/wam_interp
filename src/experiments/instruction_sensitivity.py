"""Balanced, descriptive comparisons of an existing native-future factorial bank.

Distances measure output sensitivity, not semantic correctness or policy use.
Seeds are repeated measurements within a state, never independent state samples.
"""
from itertools import combinations

import numpy as np


def balanced_contrasts(records):
    """Require a complete state x target x wording x seed factorial design."""
    if not records:
        raise ValueError("Empty bank")
    cells = {}
    for record in records:
        key = tuple(record[k] for k in ("state", "target", "wording", "seed"))
        if key in cells:
            raise ValueError("Duplicate factorial cell")
        cells[key] = record["id"]
    states = sorted({k[0] for k in cells})
    targets = sorted({k[1] for k in cells})
    wordings = sorted({k[2] for k in cells})
    seeds = sorted({k[3] for k in cells})
    if len(targets) != 2 or len(wordings) != 2 or len(seeds) < 2:
        raise ValueError("Require two targets, two action wordings and at least two seeds")
    expected = {(s,t,w,n) for s in states for t in targets for w in wordings for n in seeds}
    if set(cells) != expected:
        raise ValueError("Incomplete factorial bank")
    output = []
    def add(kind, state, a, b):
        output.append({"kind":kind, "state":state, "a":cells[a], "b":cells[b]})
    for s in states:
        for w in wordings:
            for n in seeds:
                add("target_word",s,(s,targets[0],w,n),(s,targets[1],w,n))
        for t in targets:
            for n in seeds:
                add("action_wording",s,(s,t,wordings[0],n),(s,t,wordings[1],n))
            for w in wordings:
                for a,b in combinations(seeds,2):
                    add("seed",s,(s,t,w,a),(s,t,w,b))
    return output


def future_distance(a, b):
    """RMSE on generated temporal slots only; require identical observed anchor."""
    a,b = np.asarray(a),np.asarray(b)
    if a.shape != b.shape or a.ndim != 5 or a.shape[2] < 2:
        raise ValueError("Expected matching B,C,T,H,W future arrays with T >= 2")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite future")
    if not np.array_equal(a[:,:,:1],b[:,:,:1]):
        raise ValueError("Observed anchor differs")
    x,y = a[:,:,1:].astype(np.float64),b[:,:,1:].astype(np.float64)
    d = x-y
    return {"latent_rmse":float(np.sqrt(np.mean(d*d))),
            "latent_max_abs":float(np.max(np.abs(d))),
            "future_exactly_equal":bool(np.array_equal(x,y))}


def semantic_pair(a, b):
    """Unclear labels cannot become positive evidence of a target switch."""
    labels = {"cream","butter","other","unclear"}
    for r in (a,b):
        if r.get("label") not in labels or not r.get("all_nine_frames_reviewed") or not r.get("notes"):
            raise ValueError("Complete explicit semantic review required")
    known = a["label"] != "unclear" and b["label"] != "unclear"
    return {"both_labels_clear":known,
            "different_clear_targets":known and a["label"] != b["label"],
            "both_follow_intended_target":a["label"]==a["target"] and b["label"]==b["target"]}
