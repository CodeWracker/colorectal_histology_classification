"""Materialize identical numeric arrays and auditable split identifiers once."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    from polygarbor import dataset
    root = Path(__file__).resolve().parent
    out = root / "cache"
    out.mkdir(exist_ok=True)
    start = time.perf_counter()
    bundle = dataset.load(root.parent / "polygarbor/data")
    manifest = dict(dataset=bundle.name, version="2.0.0", class_names=bundle.class_names,
                    split_expressions=dataset.DEFAULT_SPLITS, splits={})
    seen = {}
    duplicates = []
    for split in ("train", "val", "test"):
        pairs = list(dataset.iter_numpy(bundle[split]))
        x, y = np.stack([x for x, _ in pairs]), np.array([y for _, y in pairs])
        np.save(out / f"{split}_images.npy", x)
        np.save(out / f"{split}_labels.npy", y)
        hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in x]
        for i, digest in enumerate(hashes):
            if digest in seen:
                duplicates.append([seen[digest], [split, i]])
            seen[digest] = [split, i]
        manifest["splits"][split] = dict(n=len(y), counts=np.bincount(y).tolist(), sha256=hashes)
    manifest["exact_duplicates"] = duplicates
    manifest["prepare_seconds"] = time.perf_counter() - start
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}, indent=2))


if __name__ == "__main__":
    main()
