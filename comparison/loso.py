"""Leave-one-source-out folds over the ten filename-derived source codes."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "loso_manifest.json"
SPLIT_SEED = 20260913
VAL_FRACTION = .1
SPLITS = ("train", "val", "test")


def build_folds(groups, labels, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """Hold out each source for test; draw a class-stratified validation set from the remaining sources."""
    groups, labels = np.asarray(groups), np.asarray(labels)
    folds = {}
    for source in sorted(set(groups.tolist())):
        rng = np.random.default_rng([seed, int(source)])
        test = np.flatnonzero(groups == source)
        pool = np.flatnonzero(groups != source)
        val = []
        for c in np.unique(labels[pool]):
            members = rng.permutation(pool[labels[pool] == c])
            val.append(members[:min(len(members) - 1, max(1, round(val_fraction * len(members))))])
        val = np.sort(np.concatenate(val))
        folds[f"loso_{source}"] = dict(test_source=source, train=np.setdiff1d(pool, val), val=val, test=test)
    return folds


def global_labels():
    return np.concatenate([np.load(ROOT / "cache" / f"{s}_labels.npy") for s in SPLITS])


def load_fold(scenario, with_images=True):
    """Return {split: (images or None, labels)} in the order stored in loso_manifest.json."""
    fold = json.loads(MANIFEST.read_text())["folds"][scenario]
    labels = global_labels()
    images = np.concatenate([np.load(ROOT / "cache" / f"{s}_images.npy", mmap_mode="r") for s in SPLITS]) if with_images else None
    return {s: (images[fold[s]] if with_images else None, labels[fold[s]]) for s in SPLITS}


def main():
    sources = json.loads((ROOT / "source_manifest.json").read_text())
    names = json.loads((ROOT / "dataset_manifest.json").read_text())["class_names"]
    groups = np.array([row["source"] for row in sources["rows"]])
    labels = global_labels()
    folds = build_folds(groups, labels)
    payload = dict(split_seed=SPLIT_SEED, val_fraction=VAL_FRACTION, class_names=names,
                   index_space="concatenated cache train/val/test arrays, as in source_manifest.json rows",
                   folds={name: dict(test_source=fold["test_source"],
                                     counts={s: np.bincount(labels[fold[s]], minlength=len(names)).tolist() for s in SPLITS},
                                     sources={s: sorted(set(groups[fold[s]].tolist())) for s in SPLITS},
                                     **{s: fold[s].tolist() for s in SPLITS})
                          for name, fold in folds.items()})
    text = json.dumps(payload)
    # The folds are fixed before any grouped model is fitted; never regenerate them silently.
    if MANIFEST.exists() and MANIFEST.read_text() != text:
        raise RuntimeError(f"{MANIFEST.name} already exists with different folds")
    MANIFEST.write_text(text)
    for name, fold in payload["folds"].items():
        print(name, {s: sum(fold["counts"][s]) for s in SPLITS}, "test classes:", fold["counts"]["test"])


if __name__ == "__main__":
    main()
