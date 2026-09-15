"""Contact sheets from saved predictions: reproducible error cases, no refitting."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scenarios import perturb, CORRUPTIONS

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", default="2026-09-12")
    args = p.parse_args()
    out = ROOT / "results" / args.campaign
    out.mkdir(parents=True, exist_ok=True)
    names = json.loads((ROOT / "dataset_manifest.json").read_text())["class_names"]
    images = np.load(ROOT / "cache/test_images.npy", mmap_mode="r")
    y = np.load(ROOT / "cache/test_labels.npy")
    methods = ["polygarbor", "resnet18", "resnet18_imagenet", "yolo11n", "yolo11n_random"]
    predictions = {}
    for method in methods:
        path = ROOT / "runs" / args.campaign / f"full__{method}__seed42" / "eval/test_clean/predictions.npz"
        if path.exists():
            data = np.load(path)
            assert np.array_equal(data["y_true"], y)
            predictions[method] = data["y_pred"]
    methods = list(predictions)
    if len(methods) < 2:
        return
    correct = np.stack([predictions[m] == y for m in methods])
    groups = [("all correct", correct.all(0)), ("all wrong", (~correct).all(0))]
    for i, method in enumerate(methods):
        groups += [(f"only {method} correct", correct[i] & (correct.sum(0) == 1)),
                   (f"only {method} wrong", ~correct[i] & (correct.sum(0) == len(methods) - 1))]
    chosen, seen = [], set()
    for title, mask in groups:
        for i in np.flatnonzero(mask)[:2]:
            if int(i) not in seen:
                chosen.append((title, int(i)))
                seen.add(int(i))
    fig, axes = plt.subplots(int(np.ceil(len(chosen)/4)), 4, figsize=(14, 4 * int(np.ceil(len(chosen)/4))), squeeze=False, layout="constrained")
    records = []
    for ax, (title, i) in zip(axes.ravel(), chosen):
        ax.imshow(images[i])
        ax.set_title(f"#{i} true: {names[y[i]]}\n{title}", fontsize=9)
        ax.set_xlabel("\n".join(f"{m}: {names[predictions[m][i]]}" for m in methods), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        records.append(dict(index=i, selection=title, true_class=names[y[i]],
                            **{m: names[predictions[m][i]] for m in methods}))
    for ax in axes.ravel()[len(chosen):]:
        ax.axis("off")
    fig.savefig(out / "error_gallery.png", dpi=150)
    plt.close(fig)
    with (out / "error_gallery.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)

    i = 1  # predetermined first tumor, shared with explanation maps
    fig, axes = plt.subplots(3, 3, figsize=(12, 13), layout="constrained")
    for ax, kind in zip(axes.ravel(), CORRUPTIONS):
        ax.imshow(perturb(images[i], kind, i))
        ax.set_title(kind)
        lines = []
        for method in methods:
            path = ROOT / "runs" / args.campaign / f"full__{method}__seed42" / f"eval/test_{kind}/predictions.npz"
            if path.exists():
                lines.append(f"{method}: {names[np.load(path)['y_pred'][i]]}")
        ax.set_xlabel("\n".join(lines), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Same example #{i}, true class {names[y[i]]}")
    fig.savefig(out / "corruption_gallery.png", dpi=150)
    plt.close(fig)

    if not (ROOT / "source_manifest.json").exists() or not (ROOT / "cache/source_b/test_images.npy").exists():
        return
    source_manifest = json.loads((ROOT / "source_manifest.json").read_text())
    grouped = {split: (np.load(ROOT / f"cache/source_b/{split}_images.npy", mmap_mode="r"),
                       np.load(ROOT / f"cache/source_b/{split}_labels.npy")) for split in ("train", "test")}
    grouped_predictions = {}
    for method in methods:
        path = ROOT / "runs" / args.campaign / f"source_b__{method}__seed42/eval/test_clean/predictions.npz"
        if path.exists():
            grouped_predictions[method] = np.load(path)["y_pred"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 17), layout="constrained")
    for row, label in enumerate((0, 1, 6, 7)):
        for split, offset in (("train", 0), ("test", 2)):
            x, y_group = grouped[split]
            for col, i in enumerate(np.flatnonzero(y_group == label)[:2]):
                ax = axes[row, offset + col]
                global_i = source_manifest["scenarios"]["source_b"][split]["indices_global"][int(i)]
                origin = source_manifest["rows"][global_i]["source"]
                ax.imshow(x[i])
                ax.set_title(f"{split}: {names[label]} / source {origin}", fontsize=9)
                if split == "test":
                    ax.set_xlabel("\n".join(f"{m}: {names[v[i]]}" for m, v in grouped_predictions.items()), fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle("source_b: training and test examples from separate sources")
    fig.savefig(out / "source_shift_gallery.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
