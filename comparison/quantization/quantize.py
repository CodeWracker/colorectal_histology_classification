"""Export, quantize and evaluate the trained CNNs on the clean and perturbed test sets."""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TQDM_DISABLE", "1")

from common import (COMPARISON, METHODS, REPO, VARIANTS, CALIBRATION_PER_CLASS, CALIBRATION_SEED, LiteRTModel,
                    calibration_indices, export_fp32, normalize, preprocess, quantize, read_meta, source_run)

sys.path[:0] = [str(COMPARISON), str(REPO / "cnn/src"), str(REPO / "polygarbor/src")]

import numpy as np
import pandas as pd

MIN_FP32_AGREEMENT = .99


def export(run, method, out, train_x, calibration_idx):
    """Runs in its own process so TensorFlow/PyTorch memory is released before evaluation."""
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    meta = read_meta(run)
    seconds = {}
    start = time.perf_counter()
    export_fp32(method, run, out / "fp32.tflite")
    seconds["fp32"] = time.perf_counter() - start
    calibration = preprocess(method, meta, train_x[calibration_idx])
    for variant in VARIANTS[1:]:
        start = time.perf_counter()
        quantize(out / "fp32.tflite", variant, calibration, out / f"{variant}.tflite")
        seconds[variant] = time.perf_counter() - start
    (out / "export.json").write_text(json.dumps(seconds, indent=2))


def evaluate(run, method, seed, out, seconds, conditions, test_y, names, threads):
    from scenarios import CORRUPTIONS
    from cnn.evaluation import summarize
    meta = read_meta(run)
    original = json.loads((run / "summary.json").read_text())
    models = {variant: LiteRTModel(out / f"{variant}.tflite", threads) for variant in VARIANTS}
    rows = []
    # One condition at a time keeps a single preprocessed test set in memory.
    for kind in CORRUPTIONS:
        evaluation = f"test_{kind}"
        inputs = preprocess(method, meta, conditions[kind])
        reference = np.load(run / "eval" / evaluation / "predictions.npz")
        if not np.array_equal(reference["y_true"], test_y):
            raise RuntimeError(f"{run.name}: {evaluation} labels differ from the cache")
        for variant in VARIANTS:
            start = time.perf_counter()
            raw = np.stack([models[variant](inputs[i:i + 1]) for i in range(len(test_y))])
            elapsed = time.perf_counter() - start
            probabilities = normalize(raw)
            predictions = probabilities.argmax(1)
            if variant == "fp32":
                fp32_predictions = predictions
            summary, _ = summarize(test_y, probabilities, names, y_pred=predictions)
            confusion = np.zeros((len(names), len(names)), dtype=int)
            np.add.at(confusion, (test_y, predictions), 1)
            recall = np.diag(confusion) / confusion.sum(1)
            np.savez_compressed(out / "predictions" / f"{variant}__{evaluation}.npz",
                                y_true=test_y, y_pred=predictions, probabilities=probabilities)
            rows.append(dict(method=method, seed=seed, variant=variant, evaluation=evaluation, n=len(test_y),
                             tflite_bytes=(out / f"{variant}.tflite").stat().st_size,
                             conversion_seconds=seconds[variant], evaluation_seconds=elapsed,
                             accuracy=summary["accuracy"], macro_f1=summary["macro_f1"],
                             multiclass_gmean=summary["multiclass_gmean"], balanced_accuracy=summary["balanced_accuracy"],
                             original_macro_f1=original[evaluation]["macro_f1"],
                             delta_macro_f1_vs_original=summary["macro_f1"] - original[evaluation]["macro_f1"],
                             agreement_with_original=float(np.mean(predictions == reference["y_pred"])),
                             agreement_with_fp32=float(np.mean(predictions == fp32_predictions)),
                             max_raw_row_sum_error=float(np.abs(raw.sum(1) - 1).max()),
                             **{f"recall_{name}": float(value) for name, value in zip(names, recall)}))
            print(json.dumps(dict(run=out.name, variant=variant, evaluation=evaluation, macro_f1=round(summary["macro_f1"], 4),
                                  agreement_with_original=round(rows[-1]["agreement_with_original"], 4))), flush=True)
            if variant == "fp32" and kind == "clean" and rows[-1]["agreement_with_original"] < MIN_FP32_AGREEMENT:
                raise RuntimeError(f"{out.name}: fp32 conversion agrees on only {rows[-1]['agreement_with_original']:.3f} of clean test predictions")
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--export-only", action="store_true", help="internal: export and quantize, then exit")
    args = parser.parse_args()
    names = json.loads((COMPARISON / "dataset_manifest.json").read_text())["class_names"]
    cache = COMPARISON / "cache"
    train_y = np.load(cache / "train_labels.npy")
    calibration_idx = calibration_indices(train_y)
    work = COMPARISON / "runs" / args.campaign / "quantization"
    results = COMPARISON / "results" / args.campaign / "quantization"
    if args.export_only:
        train_x = np.load(cache / "train_images.npy", mmap_mode="r")
        for method in args.methods:
            for seed in args.seeds:
                export(source_run(args.campaign, method, seed), method, work / f"{method}__seed{seed}", train_x, calibration_idx)
        return

    from scenarios import CORRUPTIONS, perturb
    results.mkdir(parents=True, exist_ok=True)
    (results / "calibration.json").write_text(json.dumps(dict(
        split="train", seed=CALIBRATION_SEED, per_class=CALIBRATION_PER_CLASS, indices=calibration_idx.tolist())))
    test_x, test_y = np.load(cache / "test_images.npy"), np.load(cache / "test_labels.npy")
    conditions = {kind: test_x if kind == "clean" else np.stack([perturb(image, kind, i) for i, image in enumerate(test_x)])
                  for kind in CORRUPTIONS}
    for method in args.methods:
        for seed in args.seeds:
            out = work / f"{method}__seed{seed}"
            if (out / "metrics.csv").exists():
                print(f"SKIP {out.name}", flush=True)
                continue
            if not (out / "export.json").exists():
                subprocess.run([sys.executable, __file__, "--export-only", "--campaign", args.campaign,
                                "--methods", method, "--seeds", str(seed)], check=True)
            evaluate(source_run(args.campaign, method, seed), method, seed, out,
                     json.loads((out / "export.json").read_text()), conditions, test_y, names, args.threads)
    frames = [pd.read_csv(path) for path in sorted(work.glob("*__seed*/metrics.csv"))]
    pd.concat(frames).to_csv(results / "metrics.csv", index=False)
    print(f"Wrote {results / 'metrics.csv'} from {len(frames)} model runs")


if __name__ == "__main__":
    main()
