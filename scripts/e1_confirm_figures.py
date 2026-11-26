"""Render per-checkpoint estimates and a blinded pair-review sheet from saved stimuli."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TASK_NAMES = {5: "Book and caddy", 3: "Bowl and drawer", 7: "Soup and cheese"}


def task_name(task):
    return TASK_NAMES.get(task, str(task).replace("_", " "))


def table(result, dest):
    rows = [
        f"# {result['model']} on {result['suite']}",
        "",
        "| Task | Usable / attempted starts | Motion (mm) | Background (mm) | Selectivity (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    keys = ("motion_32_mm", "background_rms_max_mm", "selectivity_mm")
    for task in result["tasks"]:
        states = [r for r in result["states"] if r["task"] == task]
        usable = [r for r in states if r["eligible"]]
        means = [
            f"{np.mean([r['means'][k] for r in usable]):.2f}" if usable else "unavailable" for k in keys
        ]
        rows.append("| " + " | ".join([task_name(task), f"{len(usable)} / {len(states)}", *means]) + " |")
    values = []
    for k in keys:
        s = result["estimates"][k]
        if not s:
            values.append("unavailable")
            continue
        lo, hi = s["ci90" if k == "background_rms_max_mm" else "ci95"]
        values.append(f"{s['mean']:.2f} [{lo:.2f}, {hi:.2f}]")
    rows += [
        "| Equal task weights |  | " + " | ".join(values) + " |",
        "",
        "Brackets give state-stratified 95% bootstrap intervals for motion and selectivity. For background, the endpoints are one-sided 95% limits on the mean maximum trajectory RMS distance. The prespecified margin is 5 mm.",
        "",
        f"The motion generalization criterion evaluated to {result['motion_generalizes']}. The background decision was {result['background_decision']}.",
        f"The same-axis selectivity criterion evaluated to {result['same_axis_selectivity']}.",
    ]
    (dest / f"{result['model']}_table.md").write_text("\n".join(rows) + "\n", encoding="utf8")


def plot(paths, dest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for path in paths:
        result = json.loads(path.read_text())
        table(result, dest)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
        groups = result["tasks"]
        for ax, k, title, limit in zip(
            axes,
            ("motion_32_mm", "background_rms_max_mm"),
            ("Executed reach along motion axis", "Largest background response"),
            (0, 5),
        ):
            data = [
                [r["means"][k] for r in result["states"] if r["eligible"] and r["task"] == task]
                for task in groups
            ]
            ax.axhline(limit, ls="--", lw=1, color="0.35", label=f"{limit} mm reference")
            rng = np.random.default_rng(33)
            for i, g in enumerate(data):
                ax.scatter(i + rng.uniform(-0.10, 0.10, len(g)), g, s=17, alpha=0.65)
                if g:
                    ax.scatter(i, np.mean(g), marker="_", s=220, color="black")
            stats = result["estimates"][k]
            if stats:
                v = stats["mean"]
                lo, hi = stats["ci95"] if k == "motion_32_mm" else stats["ci90"]
                ax.errorbar(len(groups), v, yerr=[[v - lo], [hi - v]], fmt="D", color="black", capsize=4)
            ax.set_xticks(
                range(len(groups) + 1),
                [task_name(x) for x in groups] + ["Pooled"],
                rotation=15,
            )
            ax.set_title(title)
            ax.set_ylabel("Millimetres")
        fig.suptitle(f"{result['model']} · {result['suite']} · each dot is one start")
        fig.savefig(dest / f"{result['model']}_effects.png", dpi=180)
        plt.close(fig)


def sheet(root, out):
    spec = json.loads((root / "held.json").read_text())
    rng = np.random.default_rng(4181)
    panels = []
    for task in spec["tasks"]:
        for idx in spec["states"][str(task)]:
            folder = root / "held" / f"task_{task}/state_{idx}/seed_{spec['seeds'][0]}"
            source = folder / (spec.get("source_backend", "fastwam") + "_source.npz")
            if not source.exists():
                continue
            with np.load(source, allow_pickle=False) as clips:
                for kind, pair in (
                    ("direction", ["left_neutral", "right_neutral"]),
                    ("colour", ["right_cool", "right_warm"]),
                ):
                    rng.shuffle(pair)
                    strip = Image.new("RGB", (768, 3 * 96), "white")
                    # Each row shows both camera views. Columns are anchor, middle, and final frames.
                    for side, key in enumerate(pair):
                        if key in clips:
                            frames = clips[key]
                        else:
                            head, left, right = (
                                clips[key + "_" + c] for c in ("head", "left_wrist", "right_wrist")
                            )
                            frames = []
                            for h, l, r in zip(head, left, right):
                                bottom = np.concatenate(
                                    [np.asarray(Image.fromarray(v).resize((160, 144))) for v in (l, r)],
                                    axis=1,
                                )
                                frames.append(np.concatenate([h, bottom], axis=0))
                        for col, t in enumerate((0, len(frames) // 2, len(frames) - 1)):
                            im = Image.fromarray(frames[t])
                            im.thumbnail((256, 128), Image.Resampling.LANCZOS)
                            strip.paste(im, (256 * col + (256 - im.width) // 2, side * 140))
                    panels.append(
                        (
                            strip,
                            {
                                "task": task,
                                "state": idx,
                                "property": kind,
                                "top": pair[0],
                                "bottom": pair[1],
                            },
                        )
                    )
            break
    rng.shuffle(panels)
    if panels:
        canvas = Image.new("RGB", (768, len(panels) * 320), "white")
        draw = ImageDraw.Draw(canvas)
        key = []
        for i, (panel, answer) in enumerate(panels, 1):
            y = (i - 1) * 320
            draw.text(
                (6, y + 3), f"Pair {i}    |    initial, middle, final    |    rows A and B", fill="black"
            )
            canvas.paste(panel, (0, y + 22))
            key.append(dict(pair=i, **answer))
        canvas.save(out / "blinded_stimuli.jpg", quality=92)
        (out / "answer_key.json").write_text(json.dumps(key, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plot(sorted(a.root.glob("*_results.json")), a.out)
    if (a.root / "held.json").exists():
        sheet(a.root, a.out)


if __name__ == "__main__":
    main()
