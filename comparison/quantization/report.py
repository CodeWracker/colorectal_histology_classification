"""Tables and Pareto figure for the post-training quantization study, rebuilt from saved CSVs only."""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common import CALIBRATION_PER_CLASS, CALIBRATION_SEED, COMPARISON, METHODS, VARIANTS

sys.path.insert(0, str(COMPARISON))
from report import COLORS, markdown_table

ORDER = ("polygarbor", *METHODS)
LABELS = {"polygarbor": "PolyGabor", "resnet18": "ResNet-18 · random", "resnet18_imagenet": "ResNet-18 · ImageNet",
          "yolo11n_random": "YOLO11n · random", "yolo11n": "YOLO11n · ImageNet"}
VARIANT_ORDER = ("original", *VARIANTS)
VARIANT_LABELS = {"original": "Original (campaign runtime)", "fp32": "LiteRT float32", "fp16_weights": "LiteRT float16 weights",
                  "int8_dynamic": "LiteRT dynamic int8", "int8_static": "LiteRT static int8"}
MARKERS = {"original": "s", "fp32": "o", "fp16_weights": "^", "int8_dynamic": "D", "int8_static": "X"}


def ordered(frame, *extra):
    frame = frame.assign(_m=frame.method.map(ORDER.index), _v=frame.variant.map(VARIANT_ORDER.index))
    return frame.sort_values(["_m", "_v", *extra]).drop(columns=["_m", "_v"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    args = parser.parse_args()
    results = COMPARISON / "results" / args.campaign
    out = results / "quantization"
    metrics = pd.read_csv(out / "metrics.csv")
    campaign = pd.read_csv(results / "metrics.csv")
    edge = pd.read_csv(results / "edge.csv")
    bench = pd.read_csv(out / "benchmark.csv") if (out / "benchmark.csv").exists() else pd.DataFrame()

    clean, perturbed = metrics[metrics.evaluation == "test_clean"], metrics[metrics.evaluation != "test_clean"]
    quality = clean.groupby(["method", "variant"]).agg(
        seeds=("seed", "nunique"), size_mb=("tflite_bytes", lambda b: b.mean() / 1e6), macro_f1=("macro_f1", "mean"),
        delta_macro_f1=("delta_macro_f1_vs_original", "mean"), agreement_with_original=("agreement_with_original", "mean"),
        multiclass_gmean=("multiclass_gmean", "mean"))
    quality = quality.join(perturbed.groupby(["method", "variant"]).agg(
        perturbed_macro_f1=("macro_f1", "mean"), perturbed_delta_macro_f1=("delta_macro_f1_vs_original", "mean"),
        perturbed_agreement=("agreement_with_original", "mean"))).reset_index()
    full = campaign[(campaign.scenario == "full") & campaign.method.isin(ORDER) & campaign.evaluation.str.startswith("test_")]
    originals = []
    for method, group in full.groupby("method"):
        base, noisy = group[group.evaluation == "test_clean"], group[group.evaluation != "test_clean"]
        originals.append(dict(method=method, variant="original", seeds=base.seed.nunique(), size_mb=base.model_mb.mean(),
                              macro_f1=base.macro_f1.mean(), delta_macro_f1=0., agreement_with_original=1.,
                              multiclass_gmean=base.multiclass_gmean.mean(), perturbed_macro_f1=noisy.macro_f1.mean(),
                              perturbed_delta_macro_f1=0., perturbed_agreement=1.))
    table = ordered(pd.concat([pd.DataFrame(originals), quality], ignore_index=True))
    table.to_csv(out / "summary.csv", index=False)

    reference = edge[edge.method.isin(ORDER)].assign(variant="original")[["method", "variant", "mode", "model_mb", "median_ms", "p95_ms", "rss_peak_mb"]]
    speed = pd.concat([reference, bench[["method", "variant", "mode", "model_mb", "median_ms", "p95_ms", "invoke_median_ms",
                                         "rss_peak_mb", "agreement_with_evaluation"]] if not bench.empty else pd.DataFrame()], ignore_index=True)
    speed = ordered(speed[speed["mode"].isin(("cpu1", "cpu4"))], "mode")
    speed.to_csv(out / "speed.csv", index=False)

    panels = [("size_mb", "Model size (MB, log)", table, True)]
    for mode, field, title, log in (("cpu4", "rss_peak_mb", "Process peak RAM, cpu4 (MB)", False),
                                    ("cpu1", "median_ms", "Batch-1 latency, cpu1 (ms, log)", True)):
        chosen = speed[speed["mode"] == mode][["method", "variant", field]]
        panels.append((field, title, table[["method", "variant", "macro_f1"]].merge(chosen, on=["method", "variant"]), log))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
    for ax, (field, title, frame, log) in zip(axes, panels):
        for method in ORDER:
            points = frame[frame.method == method].sort_values(field)
            if points.empty:
                continue
            color = COLORS[method]
            ax.plot(points[field], points.macro_f1, color=color, linewidth=1.2, alpha=.5)
            for _, row in points.iterrows():
                ax.scatter(row[field], row.macro_f1, marker="*" if method == "polygarbor" else MARKERS[row.variant],
                           s=150 if method == "polygarbor" else 55, color=color, edgecolor="white", linewidth=.8, zorder=3)
            anchor = points.iloc[-1]
            ax.annotate(f" {LABELS[method]}", (anchor[field], anchor.macro_f1), fontsize=8, va="center", color="#333333")
        if log:
            ax.set_xscale("log")
        ax.set(title=title, ylim=(.7, 1))
        ax.grid(alpha=.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Clean-test macro-F1")
    handles = [Line2D([], [], marker=MARKERS[v], linestyle="", color="#7a7a7a", label=VARIANT_LABELS[v]) for v in VARIANT_ORDER]
    handles.append(Line2D([], [], marker="*", linestyle="", markersize=11, color="#7a7a7a", label="PolyGabor (full precision)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Quantized CNNs for inference versus PolyGabor")
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(out / "quantization_pareto.png", dpi=180)
    plt.close(fig)

    quality_display = table.assign(method=table.method.map(LABELS), variant=table.variant.map(VARIANT_LABELS))
    speed_display = speed.assign(method=speed.method.map(LABELS), variant=speed.variant.map(VARIANT_LABELS))
    sections = ["# Post-training quantization of the CNNs", "",
        f"Campaign `{args.campaign}`. File generated by `comparison/quantization/report.py` from `metrics.csv`, `benchmark.csv` and the campaign results. The protocol is in `comparison/PROTOCOL.md`.", "",
        "The full ResNet-18 and YOLO11n models, with random and ImageNet initialization and seeds 42 and 43, were converted to LiteRT without retraining. The Keras ResNet goes through SavedModel and the TensorFlow converter; YOLO is converted from PyTorch with litert-torch. Starting from float32, ai-edge-quantizer produces float16 weights, dynamic int8 and static int8, with float32 input and output. " +
        f"Static int8 calibration uses {8 * CALIBRATION_PER_CLASS} training crops, {CALIBRATION_PER_CLASS} per class, seed {CALIBRATION_SEED}. All variants run in the same ai-edge-litert interpreter with XNNPACK and with preprocessing that uses neither TensorFlow nor PyTorch. PolyGabor is included at full precision.", "",
        "The LiteRT float32 row separates the effect of runtime and preprocessing from the effect of quantization; the run requires at least 99% agreement with the original predictions on the clean test.", "",
        "## Performance", "", "![Pareto](quantization_pareto.png)", "",
        markdown_table(quality_display), "",
        "Means over the available seeds. `delta_macro_f1` and `agreement_with_original` compare each variant with the original model of the same seed on the clean test. The `perturbed_*` columns average the campaign's eight fixed perturbations, with the same pixels.", "",
        "## Size, latency and memory", "", markdown_table(speed_display), "",
        "Seed 42, 100 test images, a fresh process per model and mode, 10 warm-up predictions. `median_ms` includes preprocessing and execution; `invoke_median_ms` measures only the interpreter. Peak RAM is the RSS of the whole process, including Python and libraries. The original rows come from the campaign's edge benchmark, with TensorFlow, PyTorch or NumPy/OpenCV for PolyGabor, and do not isolate the runtime cost. In those rows peak RAM is sampled every 200 ms; in the LiteRT rows it is the maximum recorded by the kernel (`ru_maxrss`), which is never below the sampled value. `agreement_with_evaluation` checks the benchmark predictions against those of the evaluation.", "",
        "## Limits", "",
        "The measurements are on an x86 CPU with LiteRT; they do not represent TFLite Micro on a microcontroller, where activation memory, int8 kernels and firmware decide feasibility. Quantization is post-training, without fine-tuning. Sizes are those of the `.tflite` files, including graph metadata.", ""]
    (out / "README.md").write_text("\n".join(sections))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
