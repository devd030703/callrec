#!/usr/bin/env python3
"""Prepare a pinned, local FLEURS benchmark without extracting unsafe paths."""

import argparse
import csv
import hashlib
import io
import json
import os
import tarfile
import urllib.request
from pathlib import Path

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
DATA = {
    "hi_in": (
        "47c238b9cf7016ba596639bb3038b0c89dc6e85759f16d09c4f58c6dfa25f",
        249331325,
        "57dfd6c57689c9457b9ff1f5423438f0b9cb9ad9",
    ),
    "fr_fr": (
        "d23690e102f373554d1b544cd2ff1e76e4fedeb04953c0b72751a1b7c518cfdd",
        349036055,
        "9f5c1fd360b1bf9a73924591ede7f13f9edaf9a3",
    ),
    "en_us": (
        "d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356",
        289851356,
        "bcd7e3e4ecb8ad2a4d9754fc6be14dd6d3364b54",
    ),
}
BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_archive(path, lang, expected, size):
    if path.exists():
        if path.stat().st_size != size or digest(path) != expected:
            raise ValueError(f"Existing archive failed verification: {path}")
        return
    partial = path.with_suffix(path.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f"Partial archive is too large; retained: {partial}")
    if offset < size:
        request = urllib.request.Request(f"{BASE}/data/{lang}/audio/test.tar.gz")
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset:
                expected_range = f"bytes {offset}-{size - 1}/{size}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected_range
                ):
                    raise ValueError("Server did not honour resume range; partial file retained.")
            elif response.status != 200:
                raise ValueError(f"Unexpected HTTP status: {response.status}")
            with partial.open("ab") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
    if partial.stat().st_size != size or digest(partial) != expected:
        raise ValueError(f"Archive checksum mismatch; retained: {partial}")
    os.replace(partial, path)


def rows_for_language(lang, expected_blob):
    with urllib.request.urlopen(f"{BASE}/data/{lang}/test.tsv", timeout=30) as response:
        data = response.read()
    git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if git_hash != expected_blob:
        raise ValueError(f"Unexpected pinned transcript metadata for {lang}")
    rows = list(csv.reader(io.StringIO(data.decode("utf-8")), delimiter="\t"))
    rows.sort(key=lambda r: (int(r[0]), r[1]))
    selected = rows[:50]
    if len(selected) != 50 or len({r[1] for r in selected}) != 50:
        raise ValueError(f"Insufficient distinct cases for {lang}")
    return selected


def extract_selected(archive, destination, rows):
    wanted = {row[1] for row in rows}
    found = set()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or Path(member.name).name not in wanted:
                continue
            name = Path(member.name).name
            if name in found:
                raise ValueError(f"Duplicate audio name in archive: {name}")
            found.add(name)
            data = tar.extractfile(member).read()
            output = destination / name
            if output.exists():
                if hashlib.sha256(output.read_bytes()).digest() != hashlib.sha256(data).digest():
                    raise ValueError(f"Existing audio differs; retained: {output}")
            else:
                with output.open("xb") as stream:
                    stream.write(data)
    if found != wanted:
        raise ValueError(f"Archive missing selected files: {sorted(wanted - found)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="Private local output directory; downloads about 0.9 GB"
    )
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = []
    for lang, (expected, size, blob) in DATA.items():
        print(f"Preparing {lang}: {size / 1e6:.0f} MB archive", flush=True)
        rows = rows_for_language(lang, blob)
        archive = root / f"{lang}-test.tar.gz"
        download_archive(archive, lang, expected, size)
        extract_selected(archive, root / lang, rows)
        for index, row in enumerate(rows):
            audio = root / lang / row[1]
            manifest.append(
                {
                    "id": f"{lang}-{row[0]}-{row[1]}",
                    "language": lang,
                    "split": "development" if index < 10 else "held_out",
                    "audio": str(audio),
                    "audio_sha256": digest(audio),
                    "reference": row[2],
                    "dataset_revision": REVISION,
                }
            )
    output = root / "fleurs-manifest.json"
    content = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text() != content:
        raise ValueError(f"Existing manifest differs; retained: {output}")
    if not output.exists():
        with output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(f"Prepared {len(manifest)} cases: {output}")


if __name__ == "__main__":
    main()
