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
LABELS = {"polygarbor": "PolyGabor", "resnet18": "ResNet-18 · aleatória", "resnet18_imagenet": "ResNet-18 · ImageNet",
          "yolo11n_random": "YOLO11n · aleatória", "yolo11n": "YOLO11n · ImageNet"}
VARIANT_ORDER = ("original", *VARIANTS)
VARIANT_LABELS = {"original": "Original (runtime da campanha)", "fp32": "LiteRT float32", "fp16_weights": "LiteRT pesos float16",
                  "int8_dynamic": "LiteRT int8 dinâmico", "int8_static": "LiteRT int8 estático"}
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

    panels = [("size_mb", "Tamanho do modelo (MB, log)", table, True)]
    for mode, field, title, log in (("cpu4", "rss_peak_mb", "Pico de RAM do processo, cpu4 (MB)", False),
                                    ("cpu1", "median_ms", "Latência batch 1, cpu1 (ms, log)", True)):
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
    axes[0].set_ylabel("Macro-F1 no teste limpo")
    handles = [Line2D([], [], marker=MARKERS[v], linestyle="", color="#7a7a7a", label=VARIANT_LABELS[v]) for v in VARIANT_ORDER]
    handles.append(Line2D([], [], marker="*", linestyle="", markersize=11, color="#7a7a7a", label="PolyGabor (precisão cheia)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("CNNs quantizadas para inferência versus PolyGabor")
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(out / "quantization_pareto.png", dpi=180)
    plt.close(fig)

    quality_display = table.assign(method=table.method.map(LABELS), variant=table.variant.map(VARIANT_LABELS))
    speed_display = speed.assign(method=speed.method.map(LABELS), variant=speed.variant.map(VARIANT_LABELS))
    sections = ["# Quantização pós-treino das CNNs", "",
        f"Campanha `{args.campaign}`. Arquivo gerado por `comparison/quantization/report.py` a partir de `metrics.csv`, `benchmark.csv` e dos resultados da campanha. O protocolo está em `comparison/PROTOCOL.md`.", "",
        "Os modelos completos de ResNet-18 e YOLO11n, com inicialização aleatória e ImageNet e seeds 42 e 43, foram convertidos para LiteRT sem retreino. A ResNet Keras passa por SavedModel e pelo conversor do TensorFlow; a YOLO é convertida do PyTorch por litert-torch. A partir do float32, ai-edge-quantizer gera pesos float16, int8 dinâmico e int8 estático com entrada e saída float32. " +
        f"A calibração do int8 estático usa {8 * CALIBRATION_PER_CLASS} recortes do treino, {CALIBRATION_PER_CLASS} por classe, seed {CALIBRATION_SEED}. Todas as variantes rodam no mesmo interpretador ai-edge-litert com XNNPACK e com pré-processamento sem TensorFlow ou PyTorch. O PolyGabor entra em precisão cheia.", "",
        "A linha LiteRT float32 separa o efeito de runtime e pré-processamento do efeito da quantização; a execução exige concordância mínima de 99% com as predições originais no teste limpo.", "",
        "## Desempenho", "", "![Pareto](quantization_pareto.png)", "",
        markdown_table(quality_display), "",
        "Médias das seeds disponíveis. `delta_macro_f1` e `agreement_with_original` comparam cada variante com o modelo original da mesma seed no teste limpo. As colunas `perturbed_*` fazem a média das oito perturbações fixas da campanha, com os mesmos pixels.", "",
        "## Tamanho, latência e memória", "", markdown_table(speed_display), "",
        "Seed 42, 100 imagens de teste, processo novo por modelo e modo, 10 predições de aquecimento. `median_ms` inclui pré-processamento e execução; `invoke_median_ms` mede só o interpretador. O pico de RAM é o RSS do processo inteiro, incluindo Python e bibliotecas. As linhas originais vêm do benchmark edge da campanha, com TensorFlow, PyTorch ou NumPy/OpenCV no PolyGabor, e não isolam o custo do runtime. Nelas o pico de RAM é amostrado a cada 200 ms; nas linhas LiteRT é o máximo registrado pelo kernel (`ru_maxrss`), que nunca fica abaixo da amostragem. `agreement_with_evaluation` confere as predições do benchmark com as da avaliação.", "",
        "## Limites", "",
        "As medições são em CPU x86 com LiteRT; não representam TFLite Micro em microcontrolador, onde memória de ativações, kernels int8 e firmware decidem a viabilidade. A quantização é pós-treino, sem ajuste fino. Tamanhos são dos arquivos `.tflite`, incluindo metadados do grafo.", ""]
    (out / "README.md").write_text("\n".join(sections))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
