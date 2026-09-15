"""Rebuild tables, plots and paired statistics from saved results only."""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cnn/src"))
sys.path.insert(0, str(ROOT.parent / "polygarbor/src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from cnn.evaluation import summarize

COLORS = {"polygarbor": "#1b9e77", "resnet18": "#d95f02", "resnet18_imagenet": "#e41a1c", "yolo11n": "#7570b3", "yolo11n_random": "#377eb8", "polygarbor_p3": "#66a61e", "polygarbor_p5": "#e6ab02", "polygarbor_aug": "#e7298a", "polygarbor_p3_aug": "#a6761d"}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def markdown_table(frame):
    if frame.empty:
        return "No results yet."
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(fmt(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    args = parser.parse_args()
    root = ROOT / "runs" / args.campaign
    out = ROOT / "results" / args.campaign
    out.mkdir(parents=True, exist_ok=True)
    rows, index, runs, stages = [], [], {}, []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / "config.json").exists():
            continue
        config = read(folder / "config.json")
        status = read(folder / "status.json", {})
        entry = dict(run=folder.name, method=config["method"], scenario=config["scenario"],
                     seed=config["seed"], status=status.get("status"), error=status.get("error", ""),
                     path=str(folder.relative_to(ROOT)))
        index.append(entry)
        if entry["status"] != "complete" or config.get("pilot"):
            continue
        summaries = read(folder / "summary.json")
        timing = read(folder / "timings.json")
        resources = read(folder / "resources.json")
        for stage, elapsed in timing.items():
            stages.append(entry | dict(stage=stage) | elapsed | resources.get(stage, {}))
        selection = read(folder / "selection.json")
        size = read(folder / "model_size.json")["mb"]
        latency = read(folder / "latency.json")
        history = read(folder / "history.json", {})
        epoch_count = len(history.get("epoch_seconds", history.get("epoch", [])))
        effective = read(folder / "effective_training.json", {})
        fitted_n = sum(effective.get("fitted_vectors_per_class", {}).values()) if effective else len(selection["indices"])
        train_seconds = sum(timing.get(s, {}).get("seconds", 0) for s in
                            ("build_model", "prepare_pipeline", "extract_features", "fit"))
        retained_originals = sum(effective.get("unique_originals_retained_per_class", {}).values()) if effective.get("unique_originals_retained_per_class") else fitted_n
        base = entry | dict(n_train=len(selection["indices"]), originals_retained=retained_originals, fitted_vectors_or_images=fitted_n,
                            train_seconds=train_seconds, training_cpu_core_percent=100 * sum(timing.get(s, {}).get("cpu_seconds", 0) for s in ("build_model", "prepare_pipeline", "extract_features", "fit")) / max(train_seconds, 1e-9), epochs=epoch_count, model_mb=size,
                            batch1_median_ms=latency["median_ms"], batch1_p95_ms=latency["p95_ms"],
                            rss_peak_mb=resources["all"]["rss_mb_peak"],
                            vram_peak_mb=resources["all"]["gpu_process_mb_peak"],
                            fit_cpu_core_percent=resources.get("fit", {}).get("cpu_core_percent_mean"),
                            fit_gpu_global_percent=resources.get("fit", {}).get("gpu_global_percent_mean"))
        for evaluation, metrics in summaries.items():
            numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) or v is None}
            rows.append(base | dict(evaluation=evaluation) | numeric)
        runs[(entry["method"], entry["scenario"], entry["seed"])] = folder
    pd.DataFrame(index).to_csv(out / "run_index.csv", index=False)
    pd.DataFrame(stages).to_csv(out / "stage_resources.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics.csv", index=False)
    if frame.empty:
        print("No completed runs yet")
        return
    test = frame[frame.evaluation == "test_clean"]
    full = test[test.scenario == "full"]

    initialization_rows = []
    initialization_pairs = (("ResNet-18", "resnet18", "resnet18_imagenet"), ("YOLO11n", "yolo11n_random", "yolo11n"))
    scenario_order = ["few1", "few2", "few5", "few10", "few20", "few50", "few100", "full", "imbalance", "label_noise", "source_a", "source_b"]
    for family, random_method, imagenet_method in initialization_pairs:
        random_rows = test[test.method == random_method][["scenario", "seed", "macro_f1", "multiclass_gmean"]]
        imagenet_rows = test[test.method == imagenet_method][["scenario", "seed", "macro_f1", "multiclass_gmean"]]
        paired = random_rows.merge(imagenet_rows, on=["scenario", "seed"], suffixes=("_random", "_imagenet"))
        for scenario in scenario_order:
            group = paired[paired.scenario == scenario]
            if group.empty:
                continue
            initialization_rows.append(dict(family=family, scenario=scenario, seeds=len(group),
                macro_f1_random=group.macro_f1_random.mean(), macro_f1_imagenet=group.macro_f1_imagenet.mean(),
                macro_f1_delta_imagenet_minus_random=(group.macro_f1_imagenet - group.macro_f1_random).mean(),
                gmean_random=group.multiclass_gmean_random.mean(), gmean_imagenet=group.multiclass_gmean_imagenet.mean(),
                gmean_delta_imagenet_minus_random=(group.multiclass_gmean_imagenet - group.multiclass_gmean_random).mean()))
    initialization_frame = pd.DataFrame(initialization_rows)
    initialization_frame.to_csv(out / "initialization_comparison.csv", index=False)

    # Two seeds give a range, not a stable estimate of population variance.
    curve = test[test.scenario.str.startswith("few") | (test.scenario == "full")].copy()
    curve["examples_per_class"] = curve.scenario.map(lambda x: int(x[3:]) if x.startswith("few") else 500)
    curve.to_csv(out / "learning_curve.csv", index=False)
    curve_panels = (("Original methods", ("polygarbor", "resnet18", "yolo11n")),
                    ("CNN initialization", ("resnet18", "resnet18_imagenet", "yolo11n_random", "yolo11n")),
                    ("PolyGabor ablations", ("polygarbor", "polygarbor_aug", "polygarbor_p3", "polygarbor_p3_aug", "polygarbor_p5")))
    for metric, label, filename in (("macro_f1", "Macro-F1", "learning_curve.png"), ("multiclass_gmean", "Multiclass G-mean", "learning_curve_gmean.png")):
        fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharex=True, sharey=True)
        for ax, (title, methods) in zip(axes, curve_panels):
            for method in methods:
                group = curve[curve.method == method]
                stats = group.groupby("examples_per_class")[metric].agg(["mean", "min", "max"])
                ax.plot(stats.index, stats["mean"], "o-", label=method, color=COLORS[method], linewidth=2)
                ax.fill_between(stats.index, stats["min"], stats["max"], alpha=.10, color=COLORS[method])
            ax.set(title=title, xscale="log", ylim=(0, 1), xlabel="Original images per class")
            ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 500], labels=["1", "2", "5", "10", "20", "50", "100", "~500"])
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc="best")
        axes[0].set_ylabel(label + " on test (500 crops)")
        fig.suptitle("Performance by number of training examples")
        fig.text(.5, .01, "Lines: mean of two seeds; band: minimum–maximum. Total training = 8 × x-axis value.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
    robust = frame[(frame.scenario == "full") & frame.evaluation.str.startswith("test_")]
    pivot = robust.pivot_table(index="evaluation", columns="method", values="macro_f1")
    pivot.to_csv(out / "robustness.csv")
    if not pivot.empty:
        robust_panels = (("Methods and initialization", ("polygarbor", "resnet18", "resnet18_imagenet", "yolo11n_random", "yolo11n")),
                         ("PolyGabor ablations", ("polygarbor", "polygarbor_aug", "polygarbor_p3", "polygarbor_p3_aug", "polygarbor_p5")))
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, sharey=True)
        for ax, (title, selected) in zip(axes, robust_panels):
            columns = [method for method in selected if method in pivot.columns]
            pivot[columns].plot.bar(ax=ax, color=[COLORS[c] for c in columns], rot=30)
            ax.set(title=title, ylabel="Mean macro-F1 across seeds", xlabel="", ylim=(0, 1))
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Test condition")
        fig.tight_layout()
        fig.savefig(out / "robustness.png", dpi=180)
        plt.close(fig)
    if not full.empty:
        methods = [m for m in ("yolo11n", "resnet18_imagenet", "yolo11n_random", "resnet18", "polygarbor", "polygarbor_aug", "polygarbor_p3", "polygarbor_p3_aug", "polygarbor_p5") if m in set(full.method)]
        labels = {"yolo11n": "YOLO11n · ImageNet", "yolo11n_random": "YOLO11n · random", "resnet18": "ResNet-18 · random", "resnet18_imagenet": "ResNet-18 · ImageNet", "polygarbor": "PolyGabor", "polygarbor_aug": "PolyGabor + aug.", "polygarbor_p3": "PolyGabor · 9 regions", "polygarbor_p3_aug": "PolyGabor · 9 reg. + aug.", "polygarbor_p5": "PolyGabor · 25 regions"}
        columns = (("train_seconds", "Training (s) ↓", (15, 400), [20, 50, 100, 200]),
                   ("batch1_median_ms", "Batch-1 inference (ms) ↓", (2, 60), [2, 5, 10, 20, 50]),
                   ("model_mb", "Model (MB) ↓", (.5, 60), [.5, 1, 3, 10, 50]),
                   ("macro_f1", "Macro-F1 ↑", (.35, 1), [.4, .6, .8, 1]))
        fig, axes = plt.subplots(1, 4, figsize=(16, 6.5), sharey=True)
        for ax, (field, title, limits, ticks) in zip(axes, columns):
            for y, method in enumerate(methods):
                values = full.loc[full.method == method, field].to_numpy()
                color = COLORS[method]
                ax.hlines(y, values.min(), values.max(), color=color, linewidth=2, alpha=.5)
                ax.scatter(values, y + np.linspace(-.08, .08, len(values)), color=color, s=25, alpha=.45)
                mean = values.mean()
                ax.scatter(mean, y, color=color, edgecolor="white", linewidth=.7, marker="D", s=60, zorder=3)
                ax.annotate(f" {mean:.3g}", (mean, y), va="center", fontsize=8)
            if field != "macro_f1":
                ax.set_xscale("log")
            ax.set(title=title, xlim=limits)
            ax.set_xticks(ticks, labels=[str(x) for x in ticks])
            ax.minorticks_off()
            ax.grid(axis="x", alpha=.2)
        axes[0].set_yticks(range(len(methods)), labels=[labels[m] for m in methods])
        axes[0].invert_yaxis()
        fig.suptitle("Cost and performance with full training")
        fig.text(.5, .01, "Diamond: mean of the two seeds; dots: each seed; line: minimum–maximum.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(out / "tradeoffs.png", dpi=180)
        plt.close(fig)

    aggregation_rows = []
    for (method, scenario, seed), folder in runs.items():
        if not method.startswith("polygarbor"):
            continue
        saved = np.load(folder / "eval/test_clean/predictions.npz")
        names = read(ROOT / "dataset_manifest.json")["class_names"]
        ranked, _ = summarize(saved["y_true"], saved["probabilities"], names)
        voted = read(folder / "summary.json")["test_clean"]
        aggregation_rows.append(dict(method=method, scenario=scenario, seed=seed,
            voting_macro_f1=voted["macro_f1"], ranking_macro_f1=ranked["macro_f1"],
            voting_gmean=voted["multiclass_gmean"], ranking_gmean=ranked["multiclass_gmean"],
            agreement=float(np.mean(saved["y_pred"] == saved["probabilities"].argmax(1)))))
    aggregation_frame = pd.DataFrame(aggregation_rows)
    aggregation_frame.to_csv(out / "aggregation_comparison.csv", index=False)

    pairs = []
    source_rows = read(ROOT / "source_manifest.json")["rows"]
    source_groups = np.array([r["source"] for r in source_rows if r["original_split"] == "test"])
    for seed in sorted(full.seed.unique()):
        methods = sorted(full[full.seed == seed].method.unique())
        for a, b in itertools.combinations(methods, 2):
            pa = np.load(runs[(a, "full", seed)] / "eval/test_clean/predictions.npz")
            pb = np.load(runs[(b, "full", seed)] / "eval/test_clean/predictions.npz")
            assert np.array_equal(pa["y_true"], pb["y_true"])
            ca, cb = pa["y_pred"] == pa["y_true"], pb["y_pred"] == pb["y_true"]
            delta = cb.astype(float) - ca.astype(float)
            rng = np.random.default_rng(123)
            boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(1)
            group_names = np.unique(source_groups)
            group_sums = np.array([delta[source_groups == g].sum() for g in group_names])
            group_sizes = np.array([(source_groups == g).sum() for g in group_names])
            draws = rng.integers(0, len(group_names), size=(5000, len(group_names)))
            cluster_boot = group_sums[draws].sum(1) / group_sizes[draws].sum(1)
            only_a, only_b = int((ca & ~cb).sum()), int((cb & ~ca).sum())
            pairs.append(dict(a=a, b=b, seed=int(seed), accuracy_b_minus_a=float(delta.mean()),
                              ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
                              source_cluster_ci_low=float(np.quantile(cluster_boot, .025)),
                              source_cluster_ci_high=float(np.quantile(cluster_boot, .975)),
                              only_a=only_a, only_b=only_b,
                              mcnemar_exact_p=binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.))
    if pairs:
        order = np.argsort([p["mcnemar_exact_p"] for p in pairs])
        previous = 0
        for rank, i in enumerate(order):
            previous = max(previous, min(1., pairs[i]["mcnemar_exact_p"] * (len(pairs) - rank)))
            pairs[i]["holm_p"] = previous
    pd.DataFrame(pairs).to_csv(out / "paired_tests.csv", index=False)

    calibrated = []
    for (method, scenario, seed), folder in runs.items():
        if scenario != "full":
            continue
        val = np.load(folder / "eval/val_clean/predictions.npz")
        testp = np.load(folder / "eval/test_clean/predictions.npz")
        log_val = np.log(np.clip(val["probabilities"], 1e-12, 1))
        objective = lambda t: -np.log(np.clip(softmax(log_val / t, axis=1)[np.arange(len(val["y_true"])), val["y_true"]], 1e-12, 1)).mean()
        temperature = minimize_scalar(objective, bounds=(.05, 10), method="bounded").x
        probabilities = softmax(np.log(np.clip(testp["probabilities"], 1e-12, 1)) / temperature, axis=1)
        names = read(ROOT / "dataset_manifest.json")["class_names"]
        summary, _ = summarize(testp["y_true"], probabilities, names, y_pred=testp["y_pred"])
        calibrated.append(dict(method=method, seed=seed, temperature=float(temperature),
                               **{k: summary[k] for k in ("nll", "brier", "ece", "accuracy")}))
    pd.DataFrame(calibrated).to_csv(out / "calibration_validation_only.csv", index=False)

    edge_rows = []
    edge_root = root / "edge"
    if edge_root.exists():
        for path in sorted(edge_root.glob("*/benchmark.json")):
            if read(path.parent / "status.json", {}).get("returncode") != 0:
                continue
            cold = read(path.parent / "cold_total.json", {})
            b = read(path)
            resource = read(path.parent / "resources.json")["all"]
            times = read(path.parent / "timings.json")
            edge_rows.append(dict(method=b["method"], mode=b["mode"], median_ms=b["median_ms"],
                p95_ms=b["p95_ms"], model_mb=b["model_mb"], rss_peak_mb=resource["rss_mb_peak"],
                vram_peak_mb=resource["gpu_process_mb_peak"], load_seconds=times["imports_and_load"]["seconds"],
                first_prediction_seconds=times["first_prediction"]["seconds"],
                cold_total_seconds=cold.get("seconds") if cold.get("returncode") == 0 else None,
                batch32_images_s=b["batch_throughput_images_s"]["32"]))
    edge_frame = pd.DataFrame(edge_rows)
    edge_frame.to_csv(out / "edge.csv", index=False)
    if not edge_frame.empty:
        edge_pivot = edge_frame.pivot(index="mode", columns="method", values="median_ms")
        edge_panels = (("Methods and initialization", ("polygarbor", "resnet18", "resnet18_imagenet", "yolo11n_random", "yolo11n")),
                       ("PolyGabor ablations", ("polygarbor", "polygarbor_aug", "polygarbor_p3", "polygarbor_p3_aug", "polygarbor_p5")))
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
        for ax, (title, selected) in zip(axes, edge_panels):
            columns = [method for method in selected if method in edge_pivot.columns]
            edge_pivot[columns].plot.bar(ax=ax, color=[COLORS[c] for c in columns], rot=0)
            ax.set(title=title, ylabel="Median batch-1 latency (ms)", xlabel="")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=.2)
        axes[-1].set_xlabel("Inference environment")
        fig.tight_layout()
        fig.savefig(out / "edge.png", dpi=180)
        plt.close(fig)

    localization_path = out / "localization/metrics.csv"
    localization = pd.read_csv(localization_path) if localization_path.exists() else pd.DataFrame()
    localization_display = localization[["method", "model_seed", "test_patch_auc", "test_patch_average_precision", "test_patch_tumor_recall_at_2", "test_patch_both_tumors_top2_rate", "test_classifier_patch_auc"]].rename(columns={"method": "method", "model_seed": "seed", "test_patch_auc": "explanation AUROC", "test_patch_average_precision": "explanation AP", "test_patch_tumor_recall_at_2": "top-2 recall", "test_patch_both_tumors_top2_rate": "both in top-2", "test_classifier_patch_auc": "classifier AUROC"}) if not localization.empty else localization
    localization_section = [] if localization.empty else [
        "## Quantitative localization", "", "![Tumor patch identification](localization/localization_metrics.png)", "",
        markdown_table(localization_display), "",
        "![Effect of initialization on CAM](localization/localization_pretraining_examples.png)", "",
        "Each 4×4 mosaic contains two tumor patches and fourteen patches from the other classes, with no repetition within the split. Maps are computed on each isolated patch and reassembled without mixing between neighbors. The primary evaluation compares the explanatory evidence with the known per-patch labels; pixel-level region metrics are secondary because there is no internal annotation. Protocol, examples and limitations are in the [localization report](localization/README.md).", ""]

    sections = ["# Comparison of colorectal histopathology classifiers", "",
        f"Campaign `{args.campaign}`. This file is generated from the saved artifacts; the visual assessment and detailed interpretation are in [DISCUSSION.md](DISCUSSION.md). There are {len(index)} registered runs and {len(runs)} completed ones. See `run_index.csv` for failures and paths, `metrics.csv` for all metrics and `../../PROTOCOL.md` for the protocol.", "",
        "## Clean test after full training", "",
        markdown_table(full[["method", "seed", "accuracy", "macro_f1", "multiclass_gmean", "balanced_accuracy", "mcc", "nll", "ece"]]), "",
        "## ImageNet versus random configuration", "",
        markdown_table(initialization_frame), "",
        "Each row compares the same architecture and scenario, pairing the available seeds. A positive delta favors the ImageNet configuration. For ResNet, it combines ImageNet weights with the matching input normalization; the contrast does not isolate the weights alone. For YOLO, architecture and training policy are kept and the intended difference is the origin of the weights. The curves and the localization show that the mean gain does not hold under every source shift.", "",
        "## Full training cost", "",
        markdown_table(full[["method", "seed", "n_train", "originals_retained", "fitted_vectors_or_images", "epochs", "train_seconds", "model_mb", "rss_peak_mb", "vram_peak_mb"]]), "",
        "MB uses 1,000,000 bytes. RAM and VRAM are peaks of the whole process, including evaluation and figures; per-stage costs are in [stage_resources.csv](stage_resources.csv) and in each run's resources.json/timings.json. GPU-% is device-wide, including the desktop. CPU-% uses 100% per core. Default PolyGabor extracts 4000 vectors but fits at most 2800. In the variants, 9/25 patches and augmentation enlarge the candidate vectors while keeping the cap of 350 per class; originals_retained reports how many distinct original images reach the fit. effective_training.json details the per-class values. This changes spatial scale and retained diversity at the same time, which must be considered when interpreting the ablations.", "",
        "## CPU and GPU usage during fitting", "",
        markdown_table(full[["method", "seed", "training_cpu_core_percent", "fit_cpu_core_percent", "fit_gpu_global_percent", "vram_peak_mb"]]), "",
        "training_cpu_core_percent uses CPU-seconds divided by the total time of the training stages; 100% corresponds to one busy core. fit_cpu_core_percent is the mean sampled during fitting only. GPU-% is device-wide utilization, not exclusive to the process: desktop activity shows up even during PolyGabor. Per-PID VRAM is measured separately; PolyGabor runs no GPU operations.", "",
        "## Learning curve", "", "![Macro-F1 versus number of examples](learning_curve.png)", "",
        "![G-mean versus number of examples](learning_curve_gmean.png)", "",
        "[Interpretation differences between G-mean and macro-F1](../../GMEAN_VS_MACRO_F1.md). The x-axis always counts original images, without inflating the budget with augmentations or patches.", "",
        "The band shows the minimum and maximum across two seeds, when available; it is not a confidence interval. The ~500 point has 488–513 available examples per class, with an effective cap of 350 vectors/class in default PolyGabor. A point missing because of a fitting failure is not equivalent to zero F1. Validation keeps 500 labeled examples even in the scenarios with very few training examples; this is a training-scarcity curve, not a total annotation budget curve.", "",
        "## Test robustness", "", "![Robustness](robustness.png)", "", markdown_table(pivot.reset_index()), "",
        "The perturbations have fixed severity and identical pixel pairs across models, and do not represent external datasets or new patients. The additional source_a/source_b evaluation tests separate sources and is not mixed with these perturbations. See the parameters in scenarios.py.", "",
        "## PolyGabor spatial aggregation", "",
        markdown_table(aggregation_frame[aggregation_frame.scenario == "full"] if not aggregation_frame.empty else aggregation_frame), "",
        "Additional comparison without retraining: the main decision uses voting across regions; ranking uses the argmax of the normalized similarity of the mean distances. The main curves keep the voting fixed in the protocol. This analysis shows whether disagreement between regions explains part of the result; it was not used to retrospectively choose the best rule on the test set.", "",
        "## Paired comparisons", "", markdown_table(pd.DataFrame(pairs)), "",
        "Difference = accuracy of B minus A. Paired bootstrap with 5000 crop resamples; exact McNemar and Holm adjustment across the pairs/seeds shown. A paired bootstrap over ten source groups is also reported, resampling all crops of each source together. This interval accounts for dependence within a source and should be preferred over the crop-level one. McNemar and its Holm adjustment remain exploratory because independence between crops is not guaranteed. None of these analyses demonstrates clinical generalization.", "",
        "## Calibration", "", markdown_table(pd.DataFrame(calibrated)), "",
        "Temperature fitted exclusively on validation for each run and applied to the test set without changing the winning class. The original scores are CNN softmax and PolyGabor normalized heuristic similarity; the PolyGabor vote fraction is not used as a probability. AUROC/top-2 measure ranking; NLL/Brier/ECE need this caveat. See metrics.csv for the values before calibration.", "",
        "## Edge and inference", "", markdown_table(edge_frame), "",
        "cpu1 restricts affinity to one logical processor and libraries to one thread; cpu4 uses four threads; gpu uses the local RTX. Each measurement reloads the model in a separate process and warms up before measuring latency. Images are already in RAM: latency includes preprocessing and execution but excludes file reading and visualizations. There is no ARM emulation and no artificial RAM limit. The process approximates a computational constraint on x86; it is not a benchmark on a real edge device.", "",
        "## Generalization by source", "",
        markdown_table(test[test.scenario.str.startswith("source_")][["method", "scenario", "n_train", "accuracy", "macro_f1", "multiclass_gmean", "balanced_accuracy"]]), "",
        "Sources were recovered from the file names through SHA-256 of the pixels. source_a tests 09/10 and source_b tests 06/09; validation on 08, with the remaining sources in training. The empty class exists only in 06/10, which makes its presence in three disjoint splits at once impossible. Validation 08 contains no adipose/empty; training and test keep eight classes. The test sets differ in size/composition from the random test, so differences do not isolate the effect of source alone. Patient identity by code was not confirmed." + (" The main source analysis is now leave-one-source-out, in [loso/README.md](loso/README.md); source_a/source_b are kept as a historical record." if (out / "loso/README.md").exists() else ""), "",
        *localization_section,
        "## Artifacts and limits", "",
        "The splits are the original TFDS ones (4000/500/500) with deterministic order, auditable hashes and no exact duplicates across splits. Supervised loading exposes image/label. The audit recovered ten sources from the filenames and added source_a/source_b splits with separate sources, described in source_manifest.json. There is no clinical patient identification and no test on an external dataset. ResNet-18 and YOLO11n were run with random and ImageNet initialization; the augmentation policy and optimizer remain specific to each pipeline.", "",
        "Each run has config.json, selection.json, status.json, run.log, timings.json, resources.csv/json, the model, per-image metrics/predictions and figures. History and epochs are saved for the CNNs. Pilots live in separate campaigns and do not enter the tables. Large models and data remain on disk, ignored by Git.", "",
        "Sources: [TFDS dataset](https://www.tensorflow.org/datasets/catalog/colorectal_histology), [original data](https://zenodo.org/records/53169), [Ultralytics classification](https://docs.ultralytics.com/tasks/classify/), [activation map reference](https://keras.io/examples/vision/grad_cam/).", ""]
    (out / "REPORT.md").write_text("\n".join(sections))
    print(f"Wrote {out}; {len(runs)} completed runs")


if __name__ == "__main__":
    main()
