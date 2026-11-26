"""Record expert-accepted seeds and their task-specific active arm before policy evaluation."""

import argparse
import json
from pathlib import Path

from wam_interp.evaluations.ROBOTWIN.config import RoboTwinEvalConfig
from wam_interp.evaluations.ROBOTWIN.environment import open_environment
from wam_interp.experiments.clip_study import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--attempts", type=int, default=12)
    p.add_argument("--accepted", type=int, default=4)
    args = p.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.is_file() else []
    cfg = RoboTwinEvalConfig(
        task_name=args.task, robotwin_root=args.root.resolve(), max_repeated_screen_errors=2
    )
    with open_environment(cfg) as frontend:
        for seed in range(args.start, args.start + args.attempts):
            if sum(x["accepted"] for x in rows) >= args.accepted:
                break
            if any(x["seed"] == seed for x in rows):
                continue
            ok, info, reason = frontend.screen_seed(seed, 0)
            row = {"seed": seed, "accepted": ok, "reason": reason, "info": info}
            if ok:
                obs = frontend.begin(seed, 0, info, "unseen")
                row.update(
                    instruction=obs.instruction,
                    active_arm=str(frontend.env.arm_tag),
                    cameras={k: list(v.shape) for k, v in obs.cameras.items()},
                )
                frontend.close()
            rows.append(row)
            save(out, rows)
            if len(rows) >= 2 and reason and reason != "expert_failed" and rows[-2]["reason"] == reason:
                raise RuntimeError(reason)
    if not any(r["accepted"] for r in rows):
        raise RuntimeError(f"No expert-valid starts in {out}")


if __name__ == "__main__":
    main()
