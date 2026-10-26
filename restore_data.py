"""Restore ignored teacher data from an exact, verified upstream checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='Teacher repository checkout root')
    args = parser.parse_args()
    manifest_path = ROOT / 'upstream/code/PACKAGE_MANIFEST.json'
    if (args.source / 'code/PACKAGE_MANIFEST.json').read_bytes() != manifest_path.read_bytes():
        raise ValueError('Source package manifest differs from the recorded release')
    manifest = json.loads(manifest_path.read_text())
    files = [(relative, digest) for relative, digest in manifest['files'].items() if relative.startswith('code/data/')]
    for relative, digest in files:
        incoming = args.source / relative
        if hashlib.sha256(incoming.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Upstream data hash mismatch: {relative}')
        target = ROOT / 'upstream' / relative
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Existing local file differs; refusing to overwrite: {relative}')
    copied = 0
    for relative, _ in files:
        target = ROOT / 'upstream' / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.source / relative, target)
            copied += 1
    print(json.dumps({'verified_data_files': len(files), 'copied_files': copied}))


if __name__ == '__main__':
    main()
