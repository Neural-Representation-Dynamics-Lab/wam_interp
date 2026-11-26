"""Sequential paired E1 runner with explicit model/environment restore contract.

Importing this module requires only NumPy, not torch or a simulator.
"""

from dataclasses import dataclass, field
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class Branch:
    future: np.ndarray
    actions: np.ndarray
    commands: np.ndarray
    executed_commands: np.ndarray
    behavior: float
    metadata: dict
    trajectories: dict = field(default_factory=dict)


class Session(Protocol):
    """Implement for a loaded backend and simulator, or a CPU test double.

    snapshot/restore must include environment, controller, adapter history,
    model caches, Python/NumPy/torch CPU+CUDA RNG and explicit generators.
    run must use one fixed action chunk; behavior is a preregistered physical
    measure, never a latent coordinate or normalized action masquerading as pose.
    """
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def run(self, donor: np.ndarray | None) -> Branch: ...


def context_id(provenance):
    """Hash a fully resolved JSON context, including observation/state hashes."""
    return sha256(json.dumps(provenance, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _validate_branch(branch):
    for name in ("future", "actions", "commands", "executed_commands"):
        array = np.asarray(getattr(branch, name))
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{name} must be nonempty and finite")
    if not np.isfinite(branch.behavior):
        raise ValueError("Behavior must be finite")
    if branch.commands.ndim != 2 or branch.actions.ndim not in (2, 3):
        raise ValueError("Commands must be [time, action]; actions may include batch")
    n = len(branch.executed_commands)
    if n > len(branch.commands) or not np.array_equal(branch.executed_commands, branch.commands[:n]):
        raise ValueError("Executed commands must be an exact prefix of decoded commands")


def run_e1(session, *, donors, manifest, output):
    """Run baseline, identity, same-property and opposite-property replacements.

    Donors must already be reviewed, without selecting on policy outcomes.
    A whole-future donor swap may change nuisance content: review does not
    automatically turn this into a clean single-property causal intervention.
    Existing output directories are never overwritten. Partial runs retain a
    failed status. State is restored even when inference or serialization fails.
    """
    required = {"context", "property", "recipient_label", "expected_sign", "donors", "synthetic"}
    if not required <= manifest.keys() or manifest["expected_sign"] not in (-1, 1):
        raise ValueError("Incomplete E1 manifest or invalid expected_sign")
    if not isinstance(manifest["synthetic"], bool):
        raise ValueError("synthetic must be an explicit boolean")
    context = manifest["context"]
    fields = {"repo_revision", "backend", "checkpoint", "statistics", "resolved_config",
              "observation_sha256", "state_sha256", "prompt", "action_seed", "precision",
              "behavior_measure", "state_id"}
    if not fields <= context.keys():
        raise ValueError(f"Missing context fields: {sorted(fields - context.keys())}")
    identity = context_id(context)
    conditions = ("same_property", "opposite_property") + (("matched_same_property",) if "matched_same_property" in donors else ())
    if set(donors) != set(conditions):
        raise ValueError("Unknown donor condition")
    for name in conditions:
        review = manifest["donors"].get(name, {})
        if name not in donors or review.get("context_id") != identity:
            raise ValueError("Donors must have the same recipient context")
        array = np.asarray(donors[name])
        if not array.size or not np.isfinite(array).all():
            raise ValueError("Donors must be nonempty and finite")
        if review.get("reviewed_before_execution") is not True or not review.get("nuisance_notes"):
            raise ValueError("Donor needs independent semantic review and nuisance notes")
        same = review.get("label") == manifest["recipient_label"]
        if same != (name != "opposite_property") or review.get("label") is None:
            raise ValueError("Donor semantic label does not match its condition")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    write("manifest.json", manifest)
    np.savez_compressed(output / "donors.npz", **donors)
    write("status.json", {"status": "running", "synthetic": manifest["synthetic"]})
    try:
        snapshot = session.snapshot()
    except Exception as exc:
        write("status.json", {"status": "failed", "error": str(exc), "synthetic": manifest["synthetic"]})
        raise
    branches = {}
    try:
        for name in ("baseline", "identity", *conditions):
            session.restore(snapshot)
            donor = None if name == "baseline" else (
                branches["baseline"].future if name == "identity" else donors[name])
            branch = deepcopy(session.run(None if donor is None else np.array(donor, copy=True)))
            _validate_branch(branch)
            branches[name] = branch
            np.savez_compressed(output / f"{name}.npz", future=branch.future,
                                actions=branch.actions, commands=branch.commands,
                                executed_commands=branch.executed_commands)
            write(f"{name}.json", {"behavior": branch.behavior, "metadata": branch.metadata})
            for landmark, track in branch.trajectories.items():
                if landmark not in {"gripper", "object"}:
                    raise ValueError("Trajectory landmark must be gripper or object")
                from .trajectories import trajectory_metrics
                metrics = trajectory_metrics(track["times"], track["imagined"], track["executed"],
                    valid=track.get("valid"), executed_until=track["executed_until"])
                if not track.get("coordinate_frame") or not track.get("units"):
                    raise ValueError("Trajectories require coordinate frame and units")
                arrays = {key: track[key] for key in ("times", "imagined", "executed", "valid") if key in track}
                np.savez_compressed(output / f"{name}_{landmark}.npz", **arrays)
                write(f"{name}_{landmark}.json", {"synthetic": manifest["synthetic"],
                    "landmark": landmark, "condition": name, "state_id": context["state_id"],
                    "coordinate_frame": track["coordinate_frame"], "units": track["units"],
                    "executed_until": track["executed_until"], "metrics": metrics})
        baseline = branches["baseline"]
        if branches["identity"].actions.shape != baseline.actions.shape:
            raise ValueError("Identity control changed the action shape")
        effects = {name: float(branch.behavior - baseline.behavior) for name, branch in branches.items()}
        comparison = {key: context.get(key) for key in ("backend", "checkpoint", "statistics", "precision", "behavior_measure", "source_sha256", "base_assets")}
        comparison["property"] = manifest["property"]
        comparison["protocol"] = {key: value for key, value in context.get("protocol", {}).items()
                                  if key not in {"task_id", "initial_state_index", "environment_seed"}}
        comparison["request_options"] = {key: value for key, value in context.get("request_options", {}).items() if key != "seed"}
        summary = {"synthetic": manifest["synthetic"], "state_id": context["state_id"],
                   "comparison": comparison, "comparison_id": context_id(comparison),
                   "behavior_effects": effects,
                   "opposite_minus_same": effects["opposite_property"] - effects["same_property"],
                   "expected_sign": manifest["expected_sign"],
                   "identity_max_action_error": float(np.max(np.abs(branches["identity"].actions - baseline.actions))),
                   "interpretation": "Descriptive paired effects; semantic causality requires nuisance controls."}
        if "matched_same_property" in branches:
            summary["opposite_minus_matched_same"] = effects["opposite_property"] - effects["matched_same_property"]
        write("e1_summary.json", summary)
    except Exception as exc:
        write("status.json", {"status": "failed", "error": str(exc), "synthetic": manifest["synthetic"]})
        raise
    finally:
        try:
            session.restore(snapshot)
        except Exception as exc:
            write("status.json", {"status": "failed", "error": f"State restore failed: {exc}",
                                  "synthetic": manifest["synthetic"]})
            raise
    write("status.json", {"status": "complete", "synthetic": manifest["synthetic"]})
    return summary
