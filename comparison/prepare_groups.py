"""Recover filename-derived source groups by exact pixel hashes."""
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

# Fixed before fitting any grouped model; no performance-based selection.
ASSIGNMENTS = {
    "source_a": {"train": ["01", "02", "03", "04", "05", "06", "07"], "val": ["08"], "test": ["09", "10"]},
    "source_b": {"train": ["01", "02", "03", "04", "05", "07", "10"], "val": ["08"], "test": ["06", "09"]},
}


def main():
    os.nice(10)
    start = time.perf_counter()
    lookup = {}
    for path in (ROOT.parent / "polygarbor/data/downloads/extracted").rglob("*.tif"):
        with Image.open(path) as im:
            image = np.asarray(im.convert("RGB"))
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        source = re.search(r"CRC-Prim-HE-(\d+)", path.name).group(1)
        lookup[digest] = dict(filename=path.name, source=source)
    original = json.loads((ROOT / "dataset_manifest.json").read_text())
    images, labels, metadata = [], [], []
    for split in ("train", "val", "test"):
        x = np.load(ROOT / "cache" / f"{split}_images.npy", mmap_mode="r")
        y = np.load(ROOT / "cache" / f"{split}_labels.npy")
        images.append(x)
        labels.append(y)
        for i, digest in enumerate(original["splits"][split]["sha256"]):
            metadata.append(dict(original_split=split, original_index=i, sha256=digest, **lookup[digest]))
    x, y = np.concatenate(images), np.concatenate(labels)
    groups = np.array([r["source"] for r in metadata])
    original_overlap = {s: sorted(set(r["source"] for r in metadata if r["original_split"] == s))
                        for s in ("train", "val", "test")}
    manifest = dict(assignments=ASSIGNMENTS, original_split_sources=original_overlap, rows=metadata, scenarios={})
    for scenario, assignment in ASSIGNMENTS.items():
        assert not (set(assignment["train"]) & set(assignment["val"]) or
                    set(assignment["train"]) & set(assignment["test"]) or
                    set(assignment["val"]) & set(assignment["test"]))
        out = ROOT / "cache" / scenario
        out.mkdir(exist_ok=True)
        manifest["scenarios"][scenario] = {}
        for split, chosen_groups in assignment.items():
            indices = np.flatnonzero(np.isin(groups, chosen_groups))
            np.save(out / f"{split}_images.npy", x[indices])
            np.save(out / f"{split}_labels.npy", y[indices])
            manifest["scenarios"][scenario][split] = dict(indices_global=indices.tolist(),
                n=len(indices), counts=np.bincount(y[indices], minlength=8).tolist(), sources=chosen_groups)
    manifest["prepare_seconds"] = time.perf_counter() - start
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("rows", "scenarios")}, indent=2))


if __name__ == "__main__":
    main()
