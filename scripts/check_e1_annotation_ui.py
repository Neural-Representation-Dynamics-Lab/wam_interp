"""Optional headless-browser check of offline annotation, draft resume and export."""
import argparse
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

from wam_interp.experiments.annotation import make_annotation_page, analyze_annotations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--channel", default=None, help="Use chrome for an installed Chrome browser")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("imagined", "executed"):
        Image.new("RGB", (448, 224), (150, 180, 150)).save(args.output / f"{name}.png")
    spec = {"synthetic": True, "state_id": "ui_fixture", "condition": "baseline",
        "coordinate_frame": "agentview_rotated_cropped", "units": "normalized_image_xy",
        "pairs": [{"step": step, "imagined": "imagined.png", "executed": "executed.png"} for step in (0, 4)]}
    path = args.output / "pairs.json"
    path.write_text(json.dumps(spec))
    html = make_annotation_page(path)
    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, channel=args.channel)
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 1000}, accept_downloads=True)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(html.resolve().as_uri(), wait_until="networkidle")
            page.locator('#imagined[data-ready="true"]').wait_for()
            for side in ("imagined", "executed"):
                element = page.locator(f'#{side}[data-ready="true"]')
                box = element.bounding_box()
                assert box["width"] >= 200, "Annotation canvas is too small"
                element.click(position={"x": box["width"] / 2, "y": box["height"] / 2})
            with page.expect_download() as download:
                page.get_by_role("button", name="Save draft", exact=True).click()
            draft = args.output / "draft.json"
            download.value.save_as(draft)
            page.reload(wait_until="networkidle")
            page.locator('#load').set_input_files(draft)
            expected = json.loads(draft.read_text())["records"][0]
            page.wait_for_function("expected => JSON.stringify(JSON.parse(document.getElementById('status').textContent)) === JSON.stringify(expected)", arg=expected)
            page.locator('#landmark').select_option("object")
            page.get_by_role("button", name="Missing imagined", exact=True).click()
            page.get_by_role("button", name="Missing executed", exact=True).click()
            page.get_by_role("button", name="Next frame", exact=True).click()
            for landmark in ("gripper", "object"):
                page.locator('#landmark').select_option(landmark)
                page.get_by_role("button", name="Missing imagined", exact=True).click()
                page.get_by_role("button", name="Missing executed", exact=True).click()
            with page.expect_download() as download:
                page.get_by_role("button", name="Download completed annotations", exact=True).click()
            completed = args.output / "annotations.json"
            download.value.save_as(completed)
            page.screenshot(path=str(args.output / "annotation_page.png"), full_page=True)
            result = analyze_annotations(completed)
            assert result["metrics"]["gripper"]["coverage"] == .5
            assert result["metrics"]["gripper"]["ade"] < .005  # Browser clicks round to screen pixels.
            assert result["metrics"]["object"]["ade"] is None
            assert not errors, errors
            (args.output / "report.json").write_text(json.dumps({"passed": True, "browser_errors": errors,
                "checks": ["image load", "coordinate click", "draft download", "draft import", "missing landmarks", "complete export", "analysis"]}, indent=2))
        finally:
            browser.close()
    print(f"Annotation browser checks passed: {args.output}")


if __name__ == "__main__": main()
