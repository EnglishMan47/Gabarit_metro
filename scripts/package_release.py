"""Build an allowlisted source bundle; never include raw data or operator logs."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / 'artifacts' / 'release'
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for folder in ('webapp', 'scripts', 'tests', 'config', 'models', 'annotations', 'docs', 'ml'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.ps1', '.js', '.css', '.html', '.json', '.npz', '.onnx', '.md', '.txt'} and '__pycache__' not in path.parts:
                paths.append(path)
    paths += [ROOT / name for name in (
        'README.md', 'requirements.txt', 'requirements-dev.txt',
        'requirements-inference-cpu.txt', 'requirements-inference-gpu.txt',
        'requirements-training.txt', '.gitignore', 'FINALIZATION_PLAN.md', 'start_web.cmd'
    )]
    manifest = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    archive = destination / 'metro-lidar-source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            bundle.write(path, path.relative_to(ROOT).as_posix())
        bundle.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.sha256').write_text(f'{checksum}  {archive.name}\n', encoding='utf-8')
    print(json.dumps({'archive': str(archive), 'files': len(paths), 'bytes': archive.stat().st_size, 'sha256': checksum}, indent=2))


if __name__ == '__main__':
    main()
