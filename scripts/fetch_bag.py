"""Download a recording by link and unpack it into a bag directory (for CI runs).

Accepts a direct link or a public Yandex Disk link to a .db3 file or an archive
(.zip, .tar, .tar.gz, .zst / .tar.zst). Only *.db3 and metadata.yaml are kept.

    python scripts/fetch_bag.py https://disk.yandex.ru/d/... work/bag
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.admin import _extract_members  # noqa: E402

SQLITE_MAGIC = b"SQLite format 3\x00"


def direct_url(url: str) -> str:
    """Public Yandex Disk page -> direct download link."""
    host = urllib.parse.urlparse(url).netloc
    if "disk.yandex" in host or host.startswith("yadi.sk"):
        api = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=" + urllib.parse.quote(url, safe="")
        with urllib.request.urlopen(api, timeout=60) as r:
            return json.load(r)["href"]
    return url


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "gabarit-ci"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, step = 0, 1 << 30
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if done // step != (done - len(chunk)) // step:
                print(f"  {done / 2**30:.0f} GB" + (f" of {total / 2**30:.1f} GB" if total else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("out", type=Path, help="bag directory to create")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out.parent / (args.out.name + ".download")
    print(f"downloading {args.url}", flush=True)
    download(direct_url(args.url), tmp)
    print(f"downloaded {tmp.stat().st_size / 2**20:.0f} MB", flush=True)

    with open(tmp, "rb") as f:
        head = f.read(16)
    if head[:4] == b"\x28\xb5\x2f\xfd":                      # zstd: .zst / .tar.zst
        raw = tmp.with_suffix(".raw")
        subprocess.run(["zstd", "-d", "-f", "--rm", str(tmp), "-o", str(raw)], check=True)
        tmp = raw
        with open(tmp, "rb") as f:
            head = f.read(16)
    if head == SQLITE_MAGIC:                                  # a bare .db3
        shutil.move(tmp, args.out / f"{args.out.name}_0.db3")
    else:
        n = _extract_members(tmp, args.out)
        tmp.unlink()
        if not n:
            raise SystemExit("no .db3 file in the archive")
    for p in sorted(args.out.iterdir()):
        print(f"  {p.name}: {p.stat().st_size / 2**20:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
