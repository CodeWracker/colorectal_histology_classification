"""Fresh-process LiteRT inference benchmark, mirroring edge.py's cpu1/cpu4 modes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import COMPARISON, METHODS, VARIANTS, LiteRTModel, preprocess, read_meta, source_run


def worker(args):
    threads = 1 if args.mode == "cpu1" else 4
    if args.mode == "cpu1":
        os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    import numpy as np
    import cv2
    cv2.setNumThreads(threads)
    sys.path.insert(0, str(COMPARISON))
    from resources import Monitor
    monitor = Monitor(args.out)
    model_path = COMPARISON / "runs" / args.campaign / "quantization" / f"{args.method}__seed{args.seed}" / f"{args.variant}.tflite"
    with monitor.stage("imports_and_load"):
        meta = read_meta(source_run(args.campaign, args.method, args.seed))
        model = LiteRTModel(model_path, threads)
    images = np.array(np.load(COMPARISON / "cache/test_images.npy", mmap_mode="r")[:100], copy=True)

    def predict(image):
        return int(model(preprocess(args.method, meta, image[None])).argmax())

    with monitor.stage("first_prediction"):
        predict(images[0])
    with monitor.stage("warmup"):
        for _ in range(10):
            predict(images[0])
    latency, invoke, predictions = [], [], []
    with monitor.stage("batch1"):
        for image in images:
            start = time.perf_counter()
            x = preprocess(args.method, meta, image[None])
            ready = time.perf_counter()
            scores = model(x)
            end = time.perf_counter()
            latency.append((end - start) * 1000)
            invoke.append((end - ready) * 1000)
            predictions.append(int(scores.argmax()))
    monitor.finish()
    import resource
    result = dict(method=args.method, variant=args.variant, seed=args.seed, mode=args.mode, threads=threads,
                  # The 200 ms sampler can miss sub-second runs entirely; ru_maxrss is the kernel's high-water mark (kB).
                  max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6,
                  affinity=sorted(os.sched_getaffinity(0)), median_ms=float(np.median(latency)),
                  p95_ms=float(np.percentile(latency, 95)), invoke_median_ms=float(np.median(invoke)),
                  invoke_p95_ms=float(np.percentile(invoke, 95)), samples_ms=latency, predictions=predictions,
                  model_mb=model_path.stat().st_size / 1e6)
    (Path(args.out) / "benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples_ms", "predictions")}), flush=True)


def orchestrate(args):
    import numpy as np
    import pandas as pd
    root = COMPARISON / "runs" / args.campaign / "quantization"
    rows = []
    for method in args.methods:
        for variant in VARIANTS:
            for mode in ("cpu1", "cpu4"):
                out = root / "edge" / f"{method}__{variant}__{mode}"
                if not (out / "benchmark.json").exists():
                    out.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, __file__, "--worker", "--campaign", args.campaign, "--seed", str(args.seed),
                               "--method", method, "--variant", variant, "--mode", mode, "--out", str(out)]
                    with (out / "run.log").open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((out / "benchmark.json").read_text())
                timings = json.loads((out / "timings.json").read_text())
                reference = np.load(root / f"{method}__seed{args.seed}" / "predictions" / f"{variant}__test_clean.npz")["y_pred"][:100]
                rows.append(dict({k: result[k] for k in ("method", "variant", "seed", "mode", "threads", "median_ms", "p95_ms",
                                                          "invoke_median_ms", "invoke_p95_ms", "model_mb")},
                                 rss_peak_mb=result["max_rss_mb"],
                                 load_seconds=timings["imports_and_load"]["seconds"],
                                 first_prediction_seconds=timings["first_prediction"]["seconds"],
                                 agreement_with_evaluation=float(np.mean(np.asarray(result["predictions"]) == reference))))
                print(json.dumps(rows[-1]), flush=True)
    results = COMPARISON / "results" / args.campaign / "quantization"
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results / "benchmark.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--mode", choices=("cpu1", "cpu4"))
    parser.add_argument("--out")
    args = parser.parse_args()
    worker(args) if args.worker else orchestrate(args)


if __name__ == "__main__":
    main()
