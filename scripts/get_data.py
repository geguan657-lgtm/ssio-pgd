"""Fetch or ingest dataset archives.

The public datasets are gated behind web forms, Google Drive folders or
registration, so there is no stable URL to hard-code.  This script therefore
works from a manifest you fill in once:

    configs/download.yaml
      datasets:
        - name: euroc/MH_02_easy
          url: https://.../MH_02_easy.zip      # optional
          sha256: ...                          # optional
          dest: data/euroc/MH_02_easy

Three modes:
    python scripts/get_data.py --manifest configs/download.yaml --list
    python scripts/get_data.py --manifest configs/download.yaml --download euroc/MH_02_easy
    python scripts/get_data.py --from-dir ~/Downloads --dest data/euroc   # ingest what you already have

``--from-dir`` is the realistic path: download in a browser (which can handle
the consent forms), then let this script move, unpack and rename.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  -> {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r     {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="")
    print()
    tmp.replace(dest)
    return dest


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.name.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"unsupported archive: {archive.name}")


def ingest(src: Path, dest: Path, extract_first: bool = True) -> None:
    if src.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        print(f"  copied {src} -> {dest}")
        return
    if extract_first and src.name.endswith(ARCHIVE_SUFFIXES):
        print(f"  extracting {src.name} -> {dest}")
        extract(src, dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / src.name)
        print(f"  copied {src.name} -> {dest}")


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("datasets") or []


def main() -> int:
    p = argparse.ArgumentParser(description="Download / ingest dataset archives")
    p.add_argument("--manifest", default="configs/download.yaml")
    p.add_argument("--list", action="store_true")
    p.add_argument("--download", help="dataset name from the manifest")
    p.add_argument("--from-dir", help="folder containing archives you already downloaded")
    p.add_argument("--dest", help="destination for --from-dir")
    p.add_argument("--keep-archive", action="store_true")
    args = p.parse_args()

    entries = load_manifest(Path(args.manifest))

    if args.list or (not args.download and not args.from_dir):
        if not entries:
            print(f"No manifest at {args.manifest}.")
            print("Create one, or use --from-dir <folder-with-archives> --dest <target>.")
            return 1
        print(f"{'name':<32s}{'dest':<40s}url")
        for e in entries:
            print(f"{e.get('name', '?'):<32s}{e.get('dest', '?'):<40s}{e.get('url') or '(fill me in)'}")
        return 0

    if args.download:
        entry = next((e for e in entries if e.get("name") == args.download), None)
        if entry is None:
            print(f"unknown dataset: {args.download}", file=sys.stderr)
            return 2
        if not entry.get("url"):
            print(f"{args.download} has no url in the manifest; download it manually and use --from-dir.", file=sys.stderr)
            return 2
        dest = Path(entry["dest"])
        archive_name = entry["url"].split("/")[-1] or f"{args.download.replace('/', '_')}.zip"
        archive = dest.parent / archive_name
        download(entry["url"], archive)
        if entry.get("sha256") and sha256_of(archive) != entry["sha256"]:
            print("sha256 mismatch -- refusing to extract", file=sys.stderr)
            return 3
        extract(archive, dest)
        if not args.keep_archive:
            archive.unlink(missing_ok=True)
        print(f"done: {dest}")
        return 0

    if args.from_dir:
        if not args.dest:
            print("--from-dir needs --dest", file=sys.stderr)
            return 2
        src_dir = Path(args.from_dir).expanduser()
        if not src_dir.exists():
            print(f"missing folder: {src_dir}", file=sys.stderr)
            return 2
        items = [p for p in src_dir.iterdir() if p.suffix.lower() in (".zip", ".tar", ".gz", ".xz", ".tgz")]
        if not items:
            items = [p for p in src_dir.iterdir() if p.is_dir()]
        if not items:
            print(f"nothing to ingest in {src_dir}", file=sys.stderr)
            return 2
        print(f"ingesting {len(items)} item(s) into {args.dest}")
        for item in sorted(items):
            ingest(item, Path(args.dest) / item.stem)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())