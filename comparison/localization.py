"""Quantitative tumor localization on deterministic 4x4 patch mosaics."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cnn/src"))
sys.path.insert(0, str(ROOT.parent / "polygarbor/src"))


def make_layouts(labels, count, tumors_per_mosaic=2, seed=42):
    """Return unique patch indices and labels for deterministic 4x4 mosaics."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    positive = rng.permutation(np.flatnonzero(labels == 0))
    negative = rng.permutation(np.flatnonzero(labels != 0))
    needed_pos, needed_neg = count * tumors_per_mosaic, count * (16 - tumors_per_mosaic)
    if needed_pos > len(positive) or needed_neg > len(negative):
        raise ValueError("Not enough unique positive/negative patches for requested mosaics")
    layouts = []
    for i in range(count):
        positions = rng.choice(16, tumors_per_mosaic, replace=False)
        indices = np.empty(16, dtype=int)
        indices[positions] = positive[i * tumors_per_mosaic:(i + 1) * tumors_per_mosaic]
        other = np.setdiff1d(np.arange(16), positions)
        start = i * (16 - tumors_per_mosaic)
        indices[other] = negative[start:start + 16 - tumors_per_mosaic]
        layouts.append(dict(indices=indices.tolist(), labels=labels[indices].astype(int).tolist(), tumor_positions=sorted(map(int, positions))))
    return layouts


def build_mosaic(images, layout):
    tiles = np.asarray(images[layout["indices"]])
    side = tiles.shape[1]
    mosaic = tiles.reshape(4, 4, side, side, 3).transpose(0, 2, 1, 3, 4).reshape(4 * side, 4 * side, 3)
    tile_mask = np.asarray(layout["labels"]).reshape(4, 4) == 0
    mask = np.repeat(np.repeat(tile_mask, side, axis=0), side, axis=1)
    return mosaic, mask


def best_dice_threshold(y_true, scores):
    """Exact pooled Dice optimum, evaluating thresholds after complete ties."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    ordered_scores, ordered_true = scores[order], y_true[order]
    tp = np.cumsum(ordered_true, dtype=np.int32)
    predicted = np.arange(1, len(scores) + 1, dtype=np.int32)
    dice = 2 * tp / (predicted + int(y_true.sum()))
    ends = np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
    dice[~ends] = -1
    best = int(np.argmax(dice))
    return float(ordered_scores[best]), float(dice[best])


def dice_iou(y_true, scores, threshold):
    pred = np.asarray(scores) >= threshold
    truth = np.asarray(y_true, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    return 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp + fn, 1)


def bootstrap_mean(values, seed=123, draws=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return list(map(float, np.percentile(means, [2.5, 97.5])))


def flatten_scores(native_maps, layouts, output_size=600):
    scores, masks = [], []
    for native, layout in zip(native_maps, layouts):
        tile_mask = np.asarray(layout["labels"]).reshape(4, 4) == 0
        mask = np.repeat(np.repeat(tile_mask, output_size // 4, axis=0), output_size // 4, axis=1)
        scores.append(cv2.resize(native, (output_size, output_size), interpolation=cv2.INTER_LINEAR).ravel())
        masks.append(mask.ravel())
    return np.concatenate(masks), np.concatenate(scores)


def evaluate_maps(val_maps, test_maps, val_layouts, test_layouts):
    val_y, val_scores = flatten_scores(val_maps, val_layouts)
    threshold, val_dice = best_dice_threshold(val_y, val_scores)
    test_y, test_scores = flatten_scores(test_maps, test_layouts)
    test_dice, test_iou = dice_iou(test_y, test_scores, threshold)
    per_auc, per_ap, per_dice, per_iou = [], [], [], []
    for native, layout in zip(test_maps, test_layouts):
        y, score = flatten_scores(native[None], [layout])
        per_auc.append(roc_auc_score(y, score))
        per_ap.append(average_precision_score(y, score))
        d, i = dice_iou(y, score, threshold)
        per_dice.append(d)
        per_iou.append(i)
    return dict(threshold=threshold, validation_pooled_auc=float(roc_auc_score(val_y, val_scores)), validation_pooled_dice=val_dice,
                test_pooled_auc=float(roc_auc_score(test_y, test_scores)), test_pooled_average_precision=float(average_precision_score(test_y, test_scores)),
                test_pooled_dice=float(test_dice), test_pooled_iou=float(test_iou), test_mosaic_auc_mean=float(np.mean(per_auc)),
                test_mosaic_auc_ci95=bootstrap_mean(per_auc), test_mosaic_average_precision_mean=float(np.mean(per_ap)),
                test_mosaic_dice_mean=float(np.mean(per_dice)), test_mosaic_dice_ci95=bootstrap_mean(per_dice),
                test_mosaic_iou_mean=float(np.mean(per_iou)), per_mosaic_auc=list(map(float, per_auc)),
                per_mosaic_dice=list(map(float, per_dice)))


def polygarbor_maps(model_dir, images, layouts):
    from polygarbor import PolyGaborClassifier
    classifier = PolyGaborClassifier.load(model_dir)
    maps = []
    for layout in layouts:
        mosaic, _ = build_mosaic(images, layout)
        features, grid = classifier.extract_dense(mosaic)
        distances = classifier.distances(features)
        maps.append((-np.log1p(np.maximum(distances[:, 0], 0))).reshape(grid).astype(np.float32))
    return np.stack(maps)


def resnet_maps(model_dir, images, layouts):
    from cnn.classifier import CNNClassifier, build_resnet18, configure_tensorflow
    tf = configure_tensorflow("gpu", 4, 42)
    classifier = CNNClassifier.load(model_dir, device="gpu")
    dynamic = build_resnet18(None, classifier.n_classes)
    dynamic.set_weights(classifier.model.get_weights())
    sample = tf.image.resize(images[0], (128, 128))[None] / 255.
    max_difference = float(np.max(np.abs(classifier.model(sample, training=False).numpy() - dynamic(sample, training=False).numpy())))
    feature_model = tf.keras.Model(dynamic.inputs, dynamic.get_layer("spatial_features").output)
    weights = tf.constant(dynamic.get_layer("classifier").get_weights()[0][:, 0])

    @tf.function(input_signature=[tf.TensorSpec([1, None, None, 3], tf.float32)])
    def cam(x):
        features = feature_model(x, training=False)[0]
        return tf.nn.relu(tf.tensordot(features, weights, axes=1))

    maps = []
    for layout in layouts:
        mosaic, _ = build_mosaic(images, layout)
        maps.append(cam(tf.convert_to_tensor(mosaic[None], tf.float32) / 255.).numpy().astype(np.float32))
    return np.stack(maps), max_difference


def plot_results(result_dir, rows, saved, test_images, test_layouts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("polygarbor", "resnet18")
    labels = {"polygarbor": "PolyGabor", "resnet18": "ResNet-18 CAM"}
    colors = {"polygarbor": "#1b9e77", "resnet18": "#d95f02"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric, title in zip(axes, ("test_pooled_auc", "test_pooled_dice"), ("AUROC em pixels", "Dice no threshold de validação")):
        for x, method in enumerate(methods):
            values = [r[metric] for r in rows if r["method"] == method]
            ax.scatter([x] * len(values), values, color=colors[method], alpha=.45, s=45)
            ax.scatter(x, np.mean(values), color=colors[method], marker="D", edgecolor="white", s=80, zorder=3)
            ax.annotate(f"{np.mean(values):.3f}", (x, np.mean(values)), xytext=(7, 0), textcoords="offset points", va="center")
        ax.set(title=title, xticks=range(2), xticklabels=[labels[m] for m in methods], ylim=(0, 1))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Localização de tumor em 30 mosaicos de teste")
    fig.tight_layout()
    fig.savefig(result_dir / "localization_metrics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(14, 11), layout="constrained")
    seed_rows = {r["method"]: r for r in rows if r["model_seed"] == 42}
    for row, index in enumerate((0, 1, 2)):
        mosaic, mask = build_mosaic(test_images, test_layouts[index])
        axes[row, 0].imshow(mosaic)
        for edge in (150, 300, 450):
            axes[row, 0].axhline(edge, color="white", linewidth=.5)
            axes[row, 0].axvline(edge, color="white", linewidth=.5)
        axes[row, 0].set_title(f"Mosaico {index}")
        axes[row, 1].imshow(mosaic)
        axes[row, 1].imshow(mask, cmap="Greens", alpha=.45, vmin=0, vmax=1)
        axes[row, 1].set_title("Máscara conhecida")
        for col, method in enumerate(methods, 2):
            native = saved[(method, 42)]["test"][index]
            score = cv2.resize(native, (600, 600), interpolation=cv2.INTER_LINEAR)
            threshold = seed_rows[method]["threshold"]
            axes[row, col].imshow(mosaic)
            axes[row, col].imshow(score, cmap="magma", alpha=.55, vmin=np.percentile(saved[(method, 42)]["val"], 1), vmax=np.percentile(saved[(method, 42)]["val"], 99))
            axes[row, col].contour(score >= threshold, levels=[.5], colors=["cyan"], linewidths=1)
            y = mask.ravel()
            auc = roc_auc_score(y, score.ravel())
            dice, _ = dice_iou(y, score.ravel(), threshold)
            axes[row, col].set_title(f"{labels[method]}\nAUROC={auc:.3f}; Dice={dice:.3f}")
        for ax in axes[row]:
            ax.axis("off")
    fig.savefig(result_dir / "localization_examples.png", dpi=160)
    plt.close(fig)


def write_report(result_dir, rows, comparisons, run_dir):
    headers = ["method", "model_seed", "test_pooled_auc", "test_pooled_average_precision", "test_pooled_dice", "test_pooled_iou", "test_mosaic_auc_mean", "test_mosaic_auc_ci95", "test_mosaic_dice_mean", "test_mosaic_dice_ci95", "threshold", "map_seconds"]
    with (result_dir / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in headers} for r in rows])
    with (result_dir / "paired_comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparisons[0])
        writer.writeheader()
        writer.writerows(comparisons)
    table = ["| Método | Seed | AUROC pixels | AP pixels | Dice | IoU | AUROC médio/mosaico (IC95%) | Dice médio/mosaico (IC95%) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        table.append(f"| {r['method']} | {r['model_seed']} | {r['test_pooled_auc']:.4f} | {r['test_pooled_average_precision']:.4f} | {r['test_pooled_dice']:.4f} | {r['test_pooled_iou']:.4f} | {r['test_mosaic_auc_mean']:.4f} ({r['test_mosaic_auc_ci95'][0]:.4f}–{r['test_mosaic_auc_ci95'][1]:.4f}) | {r['test_mosaic_dice_mean']:.4f} ({r['test_mosaic_dice_ci95'][0]:.4f}–{r['test_mosaic_dice_ci95'][1]:.4f}) |")
    text = ["# Localização quantitativa de tumor em mosaicos", "", "Foram avaliados 20 mosaicos de validação e 30 de teste, todos 4×4 e 600×600 pixels, com dois patches de tumor e quatorze não tumorais. Os 800 patches utilizados são únicos dentro de cada split. Os modelos completos das seeds 42 e 43 foram reutilizados sem retreino.", "", "![Métricas de localização](localization_metrics.png)", "", *table, "", "O threshold de cada método e seed maximiza Dice somente na validação. AUROC e average precision não usam threshold. O baseline aleatório de AUROC é 0,5; marcar todos os pixels como tumor produz Dice 0,2222 nesta prevalência. Os intervalos reamostram os 30 mosaicos inteiros por 5.000 draws.", "", "![Exemplos de mosaicos, máscaras e mapas](localization_examples.png)", "", "O escore PolyGabor é `-log1p(distância para tumor)` na grade densa 75×75. O CAM da ResNet é calculado antes do softmax a partir das ativações espaciais e dos pesos da classe tumor, com ReLU. A ResNet foi estendida de forma totalmente convolucional para 600×600; a maior diferença observada em 128×128 fica registrada em raw_metrics.json. Os dois mapas foram interpolados linearmente para a resolução do mosaico.", "", "Esta avaliação mede separação espacial em uma construção sintética cujas regiões de 150×150 já possuem rótulo de classe. Ela não usa contornos celulares ou anotação de tumor dentro de uma lâmina e não valida segmentação clínica. A interpolação, o campo receptivo e as bordas entre patches afetam as métricas em pixels. A comparação pareada por mosaico está em paired_comparison.csv; mapas nativos e layouts auditáveis estão no diretório da execução.", "", f"Artefatos brutos: `{run_dir.relative_to(ROOT)}`. Tempos e recursos estão em timings.json e resources.json. A parte qualitativa nas imagens grandes não foi executada porque o arquivo local contém os 5.000 patches, sem o dataset separado `colorectal_histology_large`."]
    (result_dir / "README.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.campaign / "localization"
    result_dir = ROOT / "results" / args.campaign / "localization"
    if (run_dir / "status.json").exists() and json.loads((run_dir / "status.json").read_text()).get("status") == "complete":
        print(f"SKIP completed {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps({"status": "running"}))
    val_images = np.load(ROOT / "cache/val_images.npy", mmap_mode="r")
    val_labels = np.load(ROOT / "cache/val_labels.npy")
    test_images = np.load(ROOT / "cache/test_images.npy", mmap_mode="r")
    test_labels = np.load(ROOT / "cache/test_labels.npy")
    val_layouts = make_layouts(val_labels, 20, seed=20260912)
    test_layouts = make_layouts(test_labels, 30, seed=20260913)
    protocol = dict(grid=[4, 4], patch_pixels=[150, 150], mosaic_pixels=[600, 600], tumors_per_mosaic=2,
                    validation_mosaics=20, test_mosaics=30, unique_within_split=True, interpolation="linear",
                    threshold_selection="maximum pooled validation Dice", bootstrap_draws=5000,
                    validation_layouts=val_layouts, test_layouts=test_layouts)
    (run_dir / "layout_manifest.json").write_text(json.dumps(protocol, indent=2))
    from resources import Monitor
    monitor = Monitor(run_dir)
    saved, rows = {}, []
    try:
        for method in ("polygarbor", "resnet18"):
            for model_seed in (42, 43):
                print(f"LOCALIZATION {method} seed={model_seed}", flush=True)
                model_dir = ROOT / "runs" / args.campaign / f"full__{method}__seed{model_seed}" / "model"
                if not model_dir.exists():
                    raise FileNotFoundError(model_dir)
                started = time.perf_counter()
                with monitor.stage(f"{method}_seed{model_seed}_maps"):
                    if method == "polygarbor":
                        val_maps = polygarbor_maps(model_dir, val_images, val_layouts)
                        test_maps = polygarbor_maps(model_dir, test_images, test_layouts)
                        equivalence = None
                    else:
                        val_maps, equivalence = resnet_maps(model_dir, val_images, val_layouts)
                        test_maps, check = resnet_maps(model_dir, test_images, test_layouts)
                        equivalence = max(equivalence, check)
                map_seconds = time.perf_counter() - started
                with monitor.stage(f"{method}_seed{model_seed}_metrics"):
                    metrics = evaluate_maps(val_maps, test_maps, val_layouts, test_layouts)
                row = dict(method=method, model_seed=model_seed, map_seconds=map_seconds,
                           dynamic_128_max_probability_difference=equivalence, native_shape=list(val_maps.shape[1:]), **metrics)
                rows.append(row)
                saved[(method, model_seed)] = {"val": val_maps, "test": test_maps}
                np.savez_compressed(run_dir / f"{method}_seed{model_seed}_maps.npz", validation=val_maps, test=test_maps)
                (run_dir / "raw_metrics.json").write_text(json.dumps(rows, indent=2))
                print(f"DONE {method} seed={model_seed}: AUROC={metrics['test_pooled_auc']:.4f} Dice={metrics['test_pooled_dice']:.4f}", flush=True)
        comparisons = []
        rng = np.random.default_rng(321)
        for seed in (42, 43):
            poly = next(r for r in rows if r["method"] == "polygarbor" and r["model_seed"] == seed)
            cnn = next(r for r in rows if r["method"] == "resnet18" and r["model_seed"] == seed)
            for metric, key in (("AUROC", "per_mosaic_auc"), ("Dice", "per_mosaic_dice")):
                delta = np.asarray(poly[key]) - np.asarray(cnn[key])
                boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(1)
                comparisons.append(dict(model_seed=seed, metric=metric, polygarbor_minus_resnet=float(delta.mean()), ci95_low=float(np.percentile(boot, 2.5)), ci95_high=float(np.percentile(boot, 97.5))))
        plot_results(result_dir, rows, saved, test_images, test_layouts)
        write_report(result_dir, rows, comparisons, run_dir)
        (run_dir / "status.json").write_text(json.dumps({"status": "complete"}, indent=2))
    except Exception as exc:
        (run_dir / "status.json").write_text(json.dumps({"status": "failed", "error": repr(exc)}, indent=2))
        raise
    finally:
        monitor.finish()


if __name__ == "__main__":
    main()
