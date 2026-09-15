"""One isolated training/evaluation run with incremental artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cnn/src"))
sys.path.insert(0, str(ROOT.parent / "polygarbor/src"))


def main():
    parser = argparse.ArgumentParser()
    from variants import CNN_VARIANTS, VARIANTS
    parser.add_argument("--method", choices=[*VARIANTS, *CNN_VARIANTS], required=True)
    parser.add_argument("--scenario", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if args.method.startswith("polygarbor") or args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as np
    import cv2
    cv2.setNumThreads(args.threads)
    from resources import Monitor
    from scenarios import select_training, perturb, CORRUPTIONS
    from cnn.evaluation import save_evaluation
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists():
        raise FileExistsError(f"Run already exists: {out}; choose another directory")
    config = vars(args) | dict(python=sys.version, platform=platform.platform(),
                               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               packages={name: importlib.metadata.version(name) for name in
                                         ("numpy", "tensorflow", "torch", "torchvision", "ultralytics", "polymahalanobis", "scikit-learn")})
    config["source_hashes"] = {str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for folder in (ROOT, ROOT.parent / "cnn/src", ROOT.parent / "polygarbor/src")
                               for path in folder.rglob("*.py") if "runs" not in path.parts}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "status.json").write_text(json.dumps(dict(status="running")))
    monitor = Monitor(out)
    try:
        with monitor.stage("read_data"):
            manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
            names = manifest["class_names"]
            if args.scenario.startswith("loso_"):
                from loso import load_fold
                arrays = load_fold(args.scenario)
            else:
                cache = ROOT / "cache" / args.scenario if args.scenario.startswith("source_") else ROOT / "cache"
                arrays = {s: (np.load(cache / f"{s}_images.npy", mmap_mode="r"),
                              np.load(cache / f"{s}_labels.npy")) for s in ("train", "val", "test")}
            grouped = args.scenario.startswith(("source_", "loso_"))
            indices, labels = select_training(arrays["train"][1], "full" if grouped else args.scenario, args.seed)
            images = arrays["train"][0][indices]
            (out / "selection.json").write_text(json.dumps(dict(indices=indices.tolist(), labels=labels.tolist(),
                counts=np.bincount(labels, minlength=len(names)).tolist(), original_labels=arrays["train"][1][indices].tolist()), indent=2))
        if args.method.startswith("polygarbor"):
            from polygarbor import PolyGaborClassifier
            from variants import VARIANTS, training_views
            grid, copies = VARIANTS[args.method]
            with monitor.stage("build_model"):
                clf = PolyGaborClassifier(names, random_state=args.seed, patches_per_row=grid)
            with monitor.stage("extract_features"):
                feature_parts, vector_labels, origins = [], [], []
                for image, label, original_index in zip(images, labels, indices):
                    for view in training_views(image, copies, args.seed, original_index):
                        part = clf.extract(view)
                        feature_parts.append(part)
                        vector_labels.extend([int(label)] * len(part))
                        origins.extend([int(original_index)] * len(part))
                features = np.vstack(feature_parts)
                vector_labels, origins = np.asarray(vector_labels), np.asarray(origins)
            with monitor.stage("fit"):
                clf.fit_features(features, vector_labels)
            rng = np.random.default_rng(args.seed)
            retained = {}
            for c in range(len(names)):
                candidates = np.flatnonzero(vector_labels == c)
                if len(candidates) > clf.max_samples_per_class:
                    candidates = candidates[rng.choice(len(candidates), clf.max_samples_per_class, replace=False)]
                retained[names[c]] = len(np.unique(origins[candidates]))
            (out / "effective_training.json").write_text(json.dumps(dict(
                selected_images=len(labels), generated_training_views=len(labels) * (copies + 1),
                patches_per_row=grid, augmentation_copies=copies, augmentation="flips_brightness" if copies else "none",
                extracted_vectors=len(features), unique_originals_retained_per_class=retained,
                fitted_vectors_per_class={names[i]: len(v) for i, v in clf.train_samples.items()},
                max_samples_per_class=clf.max_samples_per_class), indent=2))
        else:
            from cnn.classifier import CNNClassifier
            architecture, weights = CNN_VARIANTS[args.method]
            clf = CNNClassifier(names, architecture=architecture, weights=weights,
                                random_state=args.seed, device=args.device, threads=args.threads)
            clf.fit(zip(images, labels), zip(*arrays["val"]), epochs=args.epochs,
                    out_dir=out, verbose=0, stage=monitor.stage)
        with monitor.stage("save_model"):
            clf.save(out / "model")
        size = sum(f.stat().st_size for f in (out / "model").rglob("*") if f.is_file())
        (out / "model_size.json").write_text(json.dumps(dict(bytes=size, mb=size / 1e6), indent=2))

        def predict(batch):
            if args.method.startswith("polygarbor"):
                predictions = [clf.predict(x) for x in batch]
                return np.stack([v.similarity for v in predictions]), np.array([v.label for v in predictions])
            scores = clf.predict_proba(batch)
            return scores, scores.argmax(1)

        with monitor.stage("warmup"):
            for _ in range(3):
                predict(images[:1])
        summaries = {}
        with monitor.stage("evaluate_train"):
            probs, pred = predict(images)
        summaries["train"] = save_evaluation(labels, probs, names, out / "eval/train", pred)
        if args.scenario == "label_noise":
            summaries["train_original_labels"] = save_evaluation(
                arrays["train"][1][indices], probs, names, out / "eval/train_original_labels", pred)
        for split in ("val", "test"):
            kinds = CORRUPTIONS if split == "test" and args.scenario == "full" and not args.pilot else ("clean",)
            for kind in kinds:
                x, y = arrays[split]
                with monitor.stage(f"prepare_{split}_{kind}"):
                    changed = [perturb(image, kind, i) for i, image in enumerate(x)]
                with monitor.stage(f"evaluate_{split}_{kind}"):
                    probs, pred = predict(changed)
                key = split + "_" + kind
                summaries[key] = save_evaluation(y, probs, names, out / "eval" / key, pred)
                print(json.dumps(dict(stage=key, accuracy=summaries[key]["accuracy"], macro_f1=summaries[key]["macro_f1"])), flush=True)
        with monitor.stage("inference_batch1"):
            latency = []
            for image in arrays["test"][0][:50]:
                t = time.perf_counter()
                predict([image])
                latency.append((time.perf_counter() - t) * 1000)
        (out / "latency.json").write_text(json.dumps(dict(batch_size=1, warmup=3, samples_ms=latency,
            median_ms=float(np.median(latency)), p95_ms=float(np.percentile(latency, 95))), indent=2))
        with monitor.stage("explanations"):
            import matplotlib.pyplot as plt
            from cnn import visualize
            visualize.use_headless()
            if args.method.startswith("resnet18"):
                fig = visualize.plot_history(clf.history)
                visualize.save_figure(fig, out / "learning_curves.png")
                plt.close(fig)
            if args.scenario == "full":
                for c in range(len(names)):
                    i = int(np.flatnonzero(arrays["test"][1] == c)[0])
                    image = arrays["test"][0][i]
                    folder = out / "explanations" / f"test_{i:03d}_{names[c]}"
                    folder.mkdir(parents=True, exist_ok=True)
                    if args.method.startswith("polygarbor"):
                        from polygarbor import visualize as pv
                        decision, dense = clf.predict(image), clf.predict(image, method="dense")
                        figs = [("decision.png", pv.plot_prediction_summary(image, decision, names, true_label=c)),
                                ("similarity.png", pv.plot_similarity_maps(image, dense, names, true_label=c))]
                        features, views = clf.extract_with_views(image)
                        figs.extend([("features.png", pv.plot_feature_matrix(features, clf.feature_names)),
                                     ("gabor.png", pv.plot_patch_decomposition(views[0], clf.bank.labels))])
                    else:
                        decision = clf.predict(image)
                        heat = visualize.activation_map(clf, image) if args.method.startswith("resnet18") else visualize.occlusion_map(clf, image)
                        np.save(folder / "heatmap.npy", heat)
                        figs = [("decision.png", visualize.plot_prediction_summary(image, decision, names, true_label=c)),
                                ("explanation.png", visualize.plot_explanation(image, heat,
                                 "CAM" if args.method.startswith("resnet18") else "Occlusion: score drop"))]
                    for filename, fig in figs:
                        visualize.save_figure(fig, folder / filename)
                        plt.close(fig)
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
        (out / "status.json").write_text(json.dumps(dict(status="complete")))
    except Exception as exc:
        (out / "status.json").write_text(json.dumps(dict(status="failed", error=repr(exc), traceback=traceback.format_exc()), indent=2))
        raise
    finally:
        monitor.finish()


if __name__ == "__main__":
    main()
