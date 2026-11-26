"""Verify an extracted source overlay before installing or running it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "HANDOFF_SOURCE.json").read_text())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != manifest["base_revision"]:
        raise ValueError(f"Expected base {manifest['base_revision']}, found {revision}")
    failures = []
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root): raise ValueError("Manifest path escapes checkout")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(name)
    if failures: raise ValueError(f"Source files missing or changed: {failures}")
    return {"verified_files": len(manifest["files"]), "base_revision": revision}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__": main()
