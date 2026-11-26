"""Freeze independent RoboTwin scene lists from official expert screening, before recipient effects."""

import argparse
import json
from pathlib import Path

from e1_confirm_analysis import read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["move_can_pot", "place_can_basket"])
    a = p.parse_args()
    base = read(Path("configs/experiments/e1_confirmation.json"))
    keys = (
        "seeds",
        "branches",
        "sham_margin_mm",
        "background_rms_margin_mm",
        "margin_basis",
        "minimum_task_coverage",
        "minimum_complete_states_per_task",
        "probe",
        "primary_background",
        "positive_motion",
        "selectivity",
    )
    common = dict(
        {k: base[k] for k in keys},
        primary_motion="Neutral RGB signed half-difference of the active TCP world-x position after 32 native controls, with a 10-control secondary prefix.",
        background="Symmetric warm and cool colour edits of the visible wall, same-physics RGB after the anchor, ramped over the first three sampled frames.",
        suite="RoboTwin2.0/demo_clean",
        tasks=a.tasks,
        source_backend="longwam",
        motion_axis="world x of expert-designated active arm",
        source_limits="20-40 mm per side, <=10 mm off-axis and passive arm displacement, <=1 mm object drift, zero robot-scene contact, elapsed times within 15%",
        planning_scope="Exploratory second simulator. Four development and at most 12 additional expert-valid scene seeds per task. Limited to this checkpoint and registered tasks.",
    )
    for split, screen, cap in (("development", "dev", 4), ("held_out", "held", 12)):
        states = {}
        for task in a.tasks:
            rows = json.loads((a.screen_dir / f"{screen}_{task}.json").read_text())
            states[task] = [r["seed"] for r in rows if r["accepted"]][:cap]
            if not states[task]:
                raise ValueError(f"No valid {task} scenes")
        write(
            a.out / f"{screen}.json", dict(common, split=split, screen_split=screen, states=states), True
        )


if __name__ == "__main__":
    main()
