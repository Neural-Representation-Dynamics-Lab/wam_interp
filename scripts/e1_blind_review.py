"""Export anonymized E1 perception and landmark checks for independent review."""
import argparse
import csv
import io
import json
import random
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerows(rows)
    return stream.getvalue()


def build(results, out, seed):
    # Stratify the visual audit across both tasks, development/held-out, and reset states.
    videos = []
    slides = results / "e1_prescribed_slide"
    for case in sorted(slides.glob("*/*/*/seed_0/source_clips.npz")):
        with np.load(case, allow_pickle=False) as data:
            videos.extend((case, name) for name in sorted(data.files))
    forecasts = sorted((results / "e1_native_replan").glob("task_*/state_*/*/idm/paired_forecast_*.jpg"))
    if not videos and not forecasts:
        raise FileNotFoundError("No forecast sheets or edit clips in the supplied results directory")
    randomizer = random.Random(seed)
    randomizer.shuffle(videos)
    randomizer.shuffle(forecasts)
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    semantic = [["clip", "moving_object", "direction_in_external_view", "robot_motion_matches_reference",
                 "visible_in_external", "visible_in_wrist", "artifacts_or_occlusions", "notes"]]
    landmarks = [["sheet", "row", "offset", "gripper_x", "gripper_y", "target_x", "target_y",
                  "gripper_visible", "target_visible", "notes"]]
    with ZipFile(out / "blind_review.zip", "w", compression=ZIP_DEFLATED) as archive:
        for index, (file, name) in enumerate(videos):
            tag = f"clip_{index:03d}"
            key[tag] = {"source": str(file.relative_to(results)), "variant": name}
            with np.load(file, allow_pickle=False) as data:
                sham, edit = data["sham"], data[name]
            order = randomizer.choice([0, 1])
            key[tag]["reference_row"] = order
            frames = []
            for a, b in zip(sham, edit, strict=True):
                # Both original camera views, no drift arrows, outcome labels or policy trajectories.
                frames.append(Image.fromarray(np.vstack((a,b) if order==0 else (b,a))))
            buf = io.BytesIO()
            frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], loop=0, duration=260)
            archive.writestr(tag+".gif", buf.getvalue())
            semantic.append([tag]+[""]*7)
        for index, file in enumerate(forecasts):
            tag = f"forecast_{index:03d}"
            with Image.open(file) as src:
                top = src.crop((0, 0, src.width, src.height//2))
                bottom = src.crop((0, src.height//2, src.width, src.height))
            if top.height != 224 or top.width not in (448,672):
                raise ValueError(f"Unexpected forecast sheet size {top.size}")
            swap = randomizer.choice([False,True])
            img = Image.fromarray(np.vstack((bottom,top) if swap else (top,bottom)))
            buf = io.BytesIO()
            img.save(buf,format="PNG")
            archive.writestr(tag+".png",buf.getvalue())
            key[tag] = {"source":str(file.relative_to(results)), "executed_row":int(swap)}
            for row in range(2):
                for offset in range(0, 4*(top.width//224),4):
                    landmarks.append([tag,row,offset]+[""]*7)
        archive.writestr("edits.csv",csv_text(semantic))
        archive.writestr("landmarks.csv",csv_text(landmarks))
        archive.writestr("instructions.txt", """Review the animations and forecast sheets independently before reading the answer key.

In each animation the two rows play together. The external camera is on the left and the wrist camera is on the right. Compare the rows, record which object visibly differs in its motion and its direction in the external view. Record whether the robot trajectory appears to agree between rows and which views allow you to tell. If both rows look the same, write 'none'. Mark uncertain judgments and occlusions.

Each forecast sheet has two rows. Left to right shows control offsets 0, 4 and, when present, 8. Row 0 is the top. Within each 224 by 224 tile, mark the gripper center between its fingers and the center of the target (the black bowl in the drawer task or the black book in the caddy task). Use local pixel coordinates with origin top left. Mark landmarks that cannot be identified as invisible and leave their coordinates blank. Keep the same anatomical reference across frames. Return both completed CSVs.
""")
    (out/"review_answer_key.json").write_text(json.dumps(key,indent=2),encoding="utf8")
    print(f"Exported {len(videos)} physical-edit clips and {len(forecasts)} forecast sheets")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--seed",type=int,default=20261122)
    args=parser.parse_args()
    build(args.results,args.output,args.seed)
