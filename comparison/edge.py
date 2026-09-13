"""Fresh-process inference benchmark with CPU affinity and model reload."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cnn/src"))
sys.path.insert(0, str(ROOT.parent / "polygarbor/src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--mode", choices=["cpu1", "cpu4", "gpu"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cold-only", action="store_true")
    args = p.parse_args()
    threads = 1 if args.mode == "cpu1" else 4
    if args.mode == "cpu1":
        os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[name] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    if args.mode != "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as np
    import cv2
    cv2.setNumThreads(threads)
    from resources import Monitor
    monitor = Monitor(args.out)
    with monitor.stage("imports_and_load"):
        if args.method.startswith("polygarbor"):
            from polygarbor import PolyGaborClassifier
            clf = PolyGaborClassifier.load(args.model_dir)
        else:
            from cnn.classifier import CNNClassifier, configure_tensorflow
            device = "gpu" if args.mode == "gpu" else "cpu"
            if args.method == "resnet18":
                configure_tensorflow(device, threads)
            else:
                import torch
                torch.set_num_threads(threads)
            clf = CNNClassifier.load(args.model_dir, device=device)
            clf.threads = threads
    images = np.array(np.load(ROOT / "cache/test_images.npy", mmap_mode="r")[:1 if args.cold_only else 100], copy=True)

    def predict(batch):
        if args.method.startswith("polygarbor"):
            return np.array([clf.predict(x).label for x in batch])
        return clf.predict_proba(batch).argmax(1)

    with monitor.stage("first_prediction"):
        first = predict(images[:1])
    if args.cold_only:
        monitor.finish()
        print(json.dumps(dict(first_prediction=first.tolist())))
        return
    if args.method == "yolo11n":
        import torch
        torch.set_num_threads(threads)
    with monitor.stage("warmup"):
        for _ in range(10):
            predict(images[:1])
    latency, predictions = [], []
    with monitor.stage("batch1"):
        for x in images:
            start = time.perf_counter()
            predictions.extend(predict([x]).tolist())
            latency.append((time.perf_counter() - start) * 1000)
    batches = {}
    for size in (8, 32):
        with monitor.stage(f"batch{size}"):
            start = time.perf_counter()
            for offset in range(0, len(images), size):
                predict(images[offset:offset + size])
            batches[str(size)] = len(images) / (time.perf_counter() - start)
    monitor.finish()
    result = dict(method=args.method, mode=args.mode, affinity=sorted(os.sched_getaffinity(0)),
                  threads=threads, effective_torch_threads=(__import__("torch").get_num_threads() if args.method == "yolo11n" else None), median_ms=float(np.median(latency)), p95_ms=float(np.percentile(latency, 95)),
                  samples_ms=latency, batch_throughput_images_s=batches, predictions=predictions,
                  model_mb=sum(p.stat().st_size for p in Path(args.model_dir).rglob("*") if p.is_file()) / 1e6)
    (Path(args.out) / "benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples_ms", "predictions")}))


if __name__ == "__main__":
    main()
