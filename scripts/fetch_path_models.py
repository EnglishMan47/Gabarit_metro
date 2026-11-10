"""Fetch pinned Attention Please weights; verify bytes before atomic replace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

REVISION = "baebf57a4169697aa61d017fcea8e2374ef135a0"
REPOSITORY = "https://github.com/BelousovArts/fod"
MODEL = {
    "name": "far_bev.pt",
    "source": "models/far_bev.pt",
    "bytes": 20_058_363,
    "git_blob_sha1": "f0b5bce8c84d9b48044cdd55ee633f0fc1778081",
}
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / ".models" / "attention_please"


def verify(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) != MODEL["bytes"]:
        raise ValueError(f"Unexpected model size: {len(raw)} bytes")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if blob != MODEL["git_blob_sha1"]:
        raise ValueError("Checkpoint identity differs from pinned upstream Git blob")
    return hashlib.sha256(raw).hexdigest()


def fetch(directory: Path = DEFAULT_DIRECTORY) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / str(MODEL["name"])
    if target.exists():
        digest = verify(target)
    else:
        url = f"https://raw.githubusercontent.com/BelousovArts/fod/{REVISION}/{MODEL['source']}"
        part = target.with_suffix(".pt.part")
        try:
            request = Request(url, headers={"User-Agent": "Gabarit-pinned-model-fetcher/1"})
            with urlopen(request, timeout=90) as response, part.open("wb") as out:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > int(MODEL["bytes"]):
                        raise ValueError("Download exceeded expected checkpoint size")
                    out.write(chunk)
            digest = verify(part)
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
    manifest = dict(MODEL, revision=REVISION, repository=REPOSITORY, sha256=digest,
                    upstream_license="not declared at pinned revision")
    target.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    try:
        path = fetch(args.directory)
    except (OSError, ValueError) as error:
        print(f"Model download failed: {error}", file=sys.stderr)
        return 1
    print(path)
    print("Upstream license is not declared; see gabarit/vendor/attention_please/NOTICE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
