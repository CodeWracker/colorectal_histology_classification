"""Data preparation of the NCT-CRC-HE-100K -> CRC-VAL-HE-7K replication.

Registered in PROTOCOL.md ("Independent-cohort replication") on 2026-09-15,
before any image was selected. The 100K archive (11.7 GB) is not downloaded:
its central directory is read with HTTP range requests and only the members
needed for validation and for the selected training images are fetched, each
with one range request, and checked against its CRC-32. The 7K archive
(0.8 GB) is downloaded whole, because every test patch is used.

Every patch is center-cropped from 224x224 to 150x150 pixels. Outputs in
cache/nct (ignored by Git): train/val/test image and label arrays and
manifest.json with class names, sources, split seeds, nested selections per
seed, file names, SHA-256 hashes of the original TIFF files and duplicate
counts.

Run from colorectal_histology_classification/ with the cnn venv:
    cnn/.venv/bin/python comparison/nct_prepare.py
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cache/nct"
URL = "https://zenodo.org/records/1214456/files/{}?download=1"
TRAIN_ZIP, TEST_ZIP = "NCT-CRC-HE-100K.zip", "CRC-VAL-HE-7K.zip"
CLASSES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
SEEDS = (42, 43, 44, 45, 46)
BUDGETS = (1, 2, 5, 10, 20)
VAL_PER_CLASS, SPLIT_SEED = 50, 20260915
SOURCE_SIZE, CROP = 224, 150
THREADS, MIN_INTERVAL = 4, 1.0  # a first run with 16 unthrottled threads was stopped by HTTP 429


class HTTPRange(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests."""

    def __init__(self, url, attempts=8):
        self.url, self.pos = url, 0
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as response:
                    self.size = int(response.headers["Content-Length"])
                return
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(10 * 2 ** attempt)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        data = fetch(self.url, self.pos, min(self.pos + len(buffer), self.size) - 1)
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


_slot_lock, _next_slot = threading.Lock(), [0.0]


def throttle():
    """Space requests by MIN_INTERVAL seconds across threads; Zenodo answers HTTP 429 to bursts."""
    with _slot_lock:
        now = time.monotonic()
        wait = _next_slot[0] - now
        _next_slot[0] = max(now, _next_slot[0]) + MIN_INTERVAL
    if wait > 0:
        time.sleep(wait)


def fetch(url, start, end, attempts=10):
    for attempt in range(attempts):
        throttle()
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise IOError(f"expected HTTP 206, got {response.status}")
                return response.read()
        except urllib.error.HTTPError as error:
            if attempt == attempts - 1:
                raise
            # Zenodo answers 429 to bursts and occasionally 502-504 when overloaded; both are retried slowly.
            transient = error.code == 429 or error.code >= 500
            retry_after = float(error.headers.get("Retry-After") or 0) if error.code == 429 else 0
            time.sleep(max(retry_after, min(10 * 2 ** attempt, 600) if transient else 2 ** attempt))
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def remote_member(url, info):
    """Bytes of one archive member, fetched with a single range request and checked by CRC-32."""
    slack = 1024
    while True:
        blob = fetch(url, info.header_offset, info.header_offset + 30 + len(info.filename.encode()) + slack + info.compress_size)
        if blob[:4] != b"PK\x03\x04":
            raise IOError(f"{info.filename}: bad local header")
        name_length, extra_length = struct.unpack("<HH", blob[26:30])
        start = 30 + name_length + extra_length
        if len(blob) >= start + info.compress_size:
            break
        slack = extra_length + 1024
    data = blob[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(data)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise IOError(f"{info.filename}: unsupported compression {info.compress_type}")
    if zlib.crc32(data) != info.CRC:
        raise IOError(f"{info.filename}: CRC mismatch")
    return data


def crop(tiff_bytes, filename):
    image = np.asarray(Image.open(io.BytesIO(tiff_bytes)).convert("RGB"))
    if image.shape != (SOURCE_SIZE, SOURCE_SIZE, 3):
        raise ValueError(f"{filename}: unexpected shape {image.shape}")
    offset = (SOURCE_SIZE - CROP) // 2
    return np.ascontiguousarray(image[offset:offset + CROP, offset:offset + CROP])


def class_of(filename):
    return CLASSES.index(filename.split("/")[-2])


def members(archive):
    return sorted((i for i in archive.infolist() if i.filename.lower().endswith(".tif")), key=lambda i: i.filename)


def download(url, path):
    if path.exists():
        return
    partial = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
        expected = int(response.headers["Content-Length"])
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    if partial.stat().st_size != expected:
        raise IOError(f"{path.name}: incomplete download")
    partial.rename(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise FileExistsError("cache/nct/manifest.json already exists; the selection is fixed and must not be redrawn")
    started = time.perf_counter()

    train_url = URL.format(TRAIN_ZIP)
    remote = HTTPRange(train_url)
    pool_members = members(zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)))
    by_class = [[i for i in pool_members if class_of(i.filename) == c] for c in range(len(CLASSES))]
    val_members, pools = [], []
    for c, files in enumerate(by_class):
        order = np.random.default_rng([SPLIT_SEED, c]).permutation(len(files))
        val_members += [files[i] for i in order[:VAL_PER_CLASS]]
        pools.append([files[i] for i in order[VAL_PER_CLASS:]])
    chosen = {seed: [[pool[i] for i in np.random.default_rng([seed, c]).permutation(len(pool))[:max(BUDGETS)]]
                     for c, pool in enumerate(pools)] for seed in SEEDS}
    train_members = sorted({i.filename: i for seed in SEEDS for files in chosen[seed] for i in files}.values(), key=lambda i: i.filename)
    print(json.dumps(dict(stage="selected", pool=len(pool_members), val=len(val_members), train_unique=len(train_members))), flush=True)

    def load_remote(info):
        # Fetched members are kept on disk, so an interrupted run resumes without new requests.
        path = OUT / "members" / info.filename
        data = path.read_bytes() if path.exists() else None
        if data is None or zlib.crc32(data) != info.CRC:
            data = remote_member(train_url, info)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return info.filename, crop(data, info.filename), hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(THREADS) as executor:
        fetched = {name: (image, digest) for name, image, digest in executor.map(load_remote, train_members + val_members)}
    print(json.dumps(dict(stage="fetched_100k", members=len(fetched), seconds=round(time.perf_counter() - started))), flush=True)

    download(URL.format(TEST_ZIP), OUT / TEST_ZIP)
    with zipfile.ZipFile(OUT / TEST_ZIP) as archive:
        test_members = members(archive)
        test = [(i.filename, crop(archive.read(i), i.filename), hashlib.sha256(archive.read(i)).hexdigest()) for i in test_members]
    print(json.dumps(dict(stage="read_7k", members=len(test), seconds=round(time.perf_counter() - started))), flush=True)

    train_names = [i.filename for i in train_members]
    splits = dict(
        train=(train_names, [fetched[n][0] for n in train_names], [fetched[n][1] for n in train_names]),
        val=([i.filename for i in val_members], [fetched[i.filename][0] for i in val_members], [fetched[i.filename][1] for i in val_members]),
        test=([n for n, _, _ in test], [x for _, x, _ in test], [d for _, _, d in test]))
    position = {name: k for k, name in enumerate(train_names)}
    selections = {str(seed): {f"few{b}": sorted(position[i.filename] for files in chosen[seed] for i in files[:b]) for b in BUDGETS}
                  for seed in SEEDS}
    pixel_hashes = {split: [hashlib.sha256(x.tobytes()).hexdigest() for x in images] for split, (_, images, _) in splits.items()}
    duplicates = {f"{a}_{b}": len(set(pixel_hashes[a]) & set(pixel_hashes[b])) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    for split, (names, images, _) in splits.items():
        np.save(OUT / f"{split}_images.npy", np.stack(images))
        np.save(OUT / f"{split}_labels.npy", np.array([class_of(n) for n in names], dtype=np.int64))
    manifest = dict(
        class_names=list(CLASSES), sources=dict(train_and_val=train_url, test=URL.format(TEST_ZIP), record="https://zenodo.org/records/1214456"),
        archive_members=dict(train_pool=len(pool_members), test=len(test_members)),
        preprocessing=f"center crop {CROP}x{CROP} from {SOURCE_SIZE}x{SOURCE_SIZE}, RGB uint8",
        split_seed=SPLIT_SEED, val_per_class=VAL_PER_CLASS, seeds=list(SEEDS), budgets=list(BUDGETS),
        counts={split: np.bincount([class_of(n) for n in names], minlength=len(CLASSES)).tolist() for split, (names, _, _) in splits.items()},
        exact_pixel_duplicates=duplicates, selections=selections,
        files={split: [dict(name=n, sha256=d) for n, d in zip(names, digests)] for split, (names, _, digests) in splits.items()})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(dict(stage="done", counts=manifest["counts"], duplicates=duplicates, seconds=round(time.perf_counter() - started))), flush=True)


if __name__ == "__main__":
    main()
