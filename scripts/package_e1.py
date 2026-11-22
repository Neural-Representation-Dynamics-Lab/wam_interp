"""Package tracked and nonignored source changes for overlay on a GPU checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/e1_handoff/source_overlay.zip"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
    excluded = {"results", "benchmark_results", "checkpoints", "third_party", "build", "dist", "__pycache__"}
    files = []
    for value in sorted(set(paths.decode().split("\0")) - {""}):
        relative = Path(value)
        path = root / relative
        if any(part in excluded or part.startswith(".venv") for part in relative.parts): continue
        if path.suffix in {".pt", ".pth", ".safetensors", ".npz", ".npy", ".pyc", ".zip"}: continue
        if path.is_symlink() or not path.is_file(): continue
        files.append((relative.as_posix(), path))
    manifest = {"base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files},
                "usage": "Overlay onto a clean clone at base_revision; includes uncommitted source."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files: archive.write(path, name)
        archive.writestr("HANDOFF_SOURCE.json", json.dumps(manifest, indent=2))
    print(json.dumps({"archive": str(args.output.resolve()), "source_files": len(files),
                      "base_revision": manifest["base_revision"], "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__": main()
