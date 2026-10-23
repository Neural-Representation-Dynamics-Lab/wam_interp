"""Prepare the pinned legacy RoboTwin stack in a dedicated Linux Python 3.10 env.

Default behavior is read-only preflight. --install installs dependencies and
applies the pinned upstream SAPIEN/MPLib fixes. Assets are downloaded separately.
"""

import argparse
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile


def filter_requirements(text):
    # The deprecated azure metapackage does not install on modern pip. Runtime
    # instruction selection reads pre-generated local descriptions and needs no
    # Azure service; retain separately named Azure packages from upstream.
    omit = {"torch", "torchvision", "huggingface-hub", "azure", "zarr"}
    result = []
    for line in text.splitlines():
        stripped = line.strip()
        name = re.split(r"[<>=!~;\[\s]", stripped, maxsplit=1)[0].replace("_", "-").lower()
        if name not in omit:
            result.append(line)
    result += ["numpy==1.26.4", "huggingface-hub==0.36.2", "zarr==2.18.3", "numcodecs==0.13.1"]
    return "\n".join(result) + "\n"


def installer_body(source):
    line = "pip install -r script/requirements.txt"
    if source.count(line) != 1:
        raise ValueError("Pinned RoboTwin installer changed: requirements installation line not found")
    source = source.replace(line, "# Dependencies installed by prepare_robotwin.py")
    clone = "git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git"
    if source.count(clone) != 1:
        raise ValueError("Pinned RoboTwin installer changed: CuRobo clone line not found")
    return "set -euo pipefail\n" + source.replace(clone, f"if [ ! -d curobo/.git ]; then {clone}; fi")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    from wam_interp.evaluations.provenance import checkout_revision
    from wam_interp.evaluations.ROBOTWIN.config import ROBOTWIN_REVISION
    root = args.root.expanduser().resolve()
    checkout_revision(root, ROBOTWIN_REVISION)
    requirements = filter_requirements((root / "script/requirements.txt").read_text())
    installer = installer_body((root / "script/_install.sh").read_text())
    print(f"Verified RoboTwin {ROBOTWIN_REVISION} at {root}")
    if not args.install:
        print("Preflight only; use --install in a fresh Linux Python 3.10 environment")
        return
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        parser.error("RoboTwin setup targets a separate Linux Python 3.10 environment")
    def pip(*arguments):
        subprocess.run([sys.executable, "-m", "pip", *arguments], check=True)
    pip("install", "torch==2.7.1", "torchvision==0.22.1", "--index-url", "https://download.pytorch.org/whl/cu128")
    # Keep these pins active during upstream native-extension installation too.
    with tempfile.TemporaryDirectory(prefix="wam-robotwin-") as directory:
        directory = Path(directory)
        constraints = directory / "constraints.txt"
        constraints.write_text("torch==2.7.1\ntorchvision==0.22.1\nhuggingface-hub==0.36.2\nnumpy==1.26.4\nzarr==2.18.3\n")
        req = directory / "requirements.txt"
        req.write_text(requirements)
        env = dict(os.environ, PIP_CONSTRAINT=str(constraints))
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], env=env, check=True)
        subprocess.run(["bash", "-c", installer], cwd=root, env=env, check=True)
    print("Environment prepared. Download assets and install the wam_interp model extras next.")


if __name__ == "__main__":
    main()
