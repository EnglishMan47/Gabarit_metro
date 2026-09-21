"""Package the current review materials without raw data or operator logs."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts" / "release"
FILES = (
    "artifacts/release/metro-lidar-source.zip",
    "artifacts/release/metro-lidar-official-template-draft.pptx",
    "artifacts/release/web-interface-demo.webm",
    "artifacts/release/positive-frame.png",
    "artifacts/release/web-demo-final.png",
    "artifacts/evaluation/verified-summary.json",
    "artifacts/evaluation/heldout-summary.json",
    "artifacts/evaluation/release-http.json",
    "docs/RELEASE_READINESS.md",
)


def main() -> None:
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Missing review files: {missing}")
    archive = RELEASE / "metro-lidar-review-package.zip"
    checksums = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in FILES:
            bundle.write(ROOT / name, name)
        bundle.writestr("MANIFEST.json", json.dumps(checksums, ensure_ascii=False, indent=2))
        bundle.writestr(
            "README_REVIEW.txt",
            "ЛЦТ 2026, задача №5 — пакет для проверки, не финальная сдача.\n"
            "Презентация использует официальный шаблон; сведения о команде пока пусты.\n"
            "Датасет, виртуальное окружение и журналы оператора не включены.\n"
            "Открытые ограничения перечислены в docs/RELEASE_READINESS.md.\n",
        )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
