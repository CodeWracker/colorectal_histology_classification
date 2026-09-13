"""Quantitative tumor localization on deterministic 4x4 patch mosaics."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cnn/src"))
sys.path.insert(0, str(ROOT.parent / "polygarbor/src"))

PROTOCOL_VERSION = 3
GRID_SIZE = 4
TILE_SIZE = 150
METHODS = ("polygarbor", "resnet18", "resnet18_imagenet")


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


def patch_scores_from_map(score):
    """Mean explanation evidence in each mosaic tile, in row-major order."""
    score = np.asarray(score)
    if score.ndim != 2 or score.shape[0] != score.shape[1] or score.shape[0] % GRID_SIZE:
        raise ValueError("score must be a square 2D mosaic divisible by the grid size")
    side = score.shape[0] // GRID_SIZE
    return score.reshape(GRID_SIZE, side, GRID_SIZE, side).mean(axis=(1, 3)).ravel()


def flatten_patch_scores(maps, layouts):
    labels = np.concatenate([np.asarray(layout["labels"]) == 0 for layout in layouts])
    scores = np.concatenate([patch_scores_from_map(score) for score in maps])
    return labels, scores


def retrieval_metrics(scores, layouts):
    recalls, exact = [], []
    for score, layout in zip(scores, layouts):
        truth = np.asarray(layout["labels"]) == 0
        top = np.argsort(-np.asarray(score), kind="stable")[:int(truth.sum())]
        recall = float(truth[top].sum() / truth.sum())
        recalls.append(recall)
        exact.append(float(recall == 1.0))
    return recalls, exact


def evaluate_maps(val_maps, test_maps, val_layouts, test_layouts, val_classifier_scores, test_classifier_scores):
    """Evaluate weak regions, patch evidence, and the classifier's tumor score."""
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
    val_patch_y, val_patch_scores = flatten_patch_scores(val_maps, val_layouts)
    patch_threshold, validation_patch_dice = best_dice_threshold(val_patch_y, val_patch_scores)
    test_patch_y, test_patch_scores = flatten_patch_scores(test_maps, test_layouts)
    test_patch_dice, test_patch_iou = dice_iou(test_patch_y, test_patch_scores, patch_threshold)
    per_patch_auc, per_patch_dice = [], []
    explanation_scores = []
    for score, layout in zip(test_maps, test_layouts):
        patch_score = patch_scores_from_map(score)
        truth = np.asarray(layout["labels"]) == 0
        explanation_scores.append(patch_score)
        per_patch_auc.append(roc_auc_score(truth, patch_score))
        per_patch_dice.append(dice_iou(truth, patch_score, patch_threshold)[0])
    explanation_recall, explanation_exact = retrieval_metrics(explanation_scores, test_layouts)
    classifier_recall, classifier_exact = retrieval_metrics(test_classifier_scores, test_layouts)
    validation_classifier_y = np.concatenate([np.asarray(layout["labels"]) == 0 for layout in val_layouts])
    validation_classifier_scores = np.asarray(val_classifier_scores).ravel()
    classifier_y = np.concatenate([np.asarray(layout["labels"]) == 0 for layout in test_layouts])
    classifier_scores = np.asarray(test_classifier_scores).ravel()
    return dict(threshold=threshold, validation_pooled_auc=float(roc_auc_score(val_y, val_scores)), validation_pooled_dice=val_dice,
                test_pooled_auc=float(roc_auc_score(test_y, test_scores)), test_pooled_average_precision=float(average_precision_score(test_y, test_scores)),
                test_pooled_dice=float(test_dice), test_pooled_iou=float(test_iou), test_mosaic_auc_mean=float(np.mean(per_auc)),
                test_mosaic_auc_ci95=bootstrap_mean(per_auc), test_mosaic_average_precision_mean=float(np.mean(per_ap)),
                test_mosaic_dice_mean=float(np.mean(per_dice)), test_mosaic_dice_ci95=bootstrap_mean(per_dice),
                test_mosaic_iou_mean=float(np.mean(per_iou)), per_mosaic_auc=list(map(float, per_auc)),
                per_mosaic_dice=list(map(float, per_dice)), patch_threshold=patch_threshold,
                validation_patch_dice=validation_patch_dice, test_patch_auc=float(roc_auc_score(test_patch_y, test_patch_scores)),
                test_patch_average_precision=float(average_precision_score(test_patch_y, test_patch_scores)),
                test_patch_dice=float(test_patch_dice), test_patch_iou=float(test_patch_iou),
                test_patch_auc_mean=float(np.mean(per_patch_auc)), test_patch_auc_ci95=bootstrap_mean(per_patch_auc),
                test_patch_dice_mean=float(np.mean(per_patch_dice)), test_patch_dice_ci95=bootstrap_mean(per_patch_dice),
                test_patch_tumor_recall_at_2=float(np.mean(explanation_recall)),
                test_patch_both_tumors_top2_rate=float(np.mean(explanation_exact)),
                test_classifier_patch_auc=float(roc_auc_score(classifier_y, classifier_scores)),
                test_classifier_patch_average_precision=float(average_precision_score(classifier_y, classifier_scores)),
                validation_classifier_patch_auc=float(roc_auc_score(validation_classifier_y, validation_classifier_scores)),
                test_classifier_tumor_recall_at_2=float(np.mean(classifier_recall)),
                test_classifier_both_tumors_top2_rate=float(np.mean(classifier_exact)),
                per_mosaic_patch_auc=list(map(float, per_patch_auc)), per_mosaic_patch_dice=list(map(float, per_patch_dice)),
                per_mosaic_patch_recall_at_2=explanation_recall)


def stitch_tile_maps(tile_maps, tile_size=TILE_SIZE):
    """Resize within each tile and stitch without interpolation across boundaries."""
    output = np.empty((GRID_SIZE * tile_size, GRID_SIZE * tile_size), dtype=np.float32)
    for position, tile_map in enumerate(tile_maps):
        row, col = divmod(position, GRID_SIZE)
        resized = cv2.resize(np.asarray(tile_map, dtype=np.float32), (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        output[row * tile_size:(row + 1) * tile_size, col * tile_size:(col + 1) * tile_size] = resized
    return output


def polygarbor_maps(model_dir, images, layouts):
    from polygarbor import PolyGaborClassifier
    classifier = PolyGaborClassifier.load(model_dir)
    maps, classifier_scores = [], []
    for layout in layouts:
        tile_maps, tile_scores = [], []
        for index in layout["indices"]:
            image = images[index]
            features, grid = classifier.extract_dense(image)
            distances = classifier.distances(features)
            tile_maps.append((-np.log1p(np.maximum(distances[:, 0], 0))).reshape(grid).astype(np.float32))
            tile_scores.append(float(classifier.predict(image).similarity[0]))
        maps.append(stitch_tile_maps(tile_maps))
        classifier_scores.append(tile_scores)
    return np.stack(maps), np.asarray(classifier_scores, dtype=np.float32), list(tile_maps[0].shape)


def resnet_maps(model_dir, images, layouts):
    from cnn.classifier import CNNClassifier, configure_tensorflow
    tf = configure_tensorflow("gpu", 4, 42)
    classifier = CNNClassifier.load(model_dir, device="gpu")
    feature_model = tf.keras.Model(classifier.model.inputs, classifier.model.get_layer("spatial_features").output)
    weights = tf.constant(classifier.model.get_layer("classifier").get_weights()[0][:, 0])

    @tf.function(input_signature=[tf.TensorSpec([None, 128, 128, 3], tf.float32)])
    def cam(x):
        features = feature_model(x, training=False)
        maps = tf.nn.relu(tf.tensordot(features, weights, axes=[[3], [0]]))
        return maps, classifier.model(x, training=False)

    maps, classifier_scores = [], []
    native_shape = None
    for layout in layouts:
        tiles = tf.image.resize(np.asarray(images[layout["indices"]]), (128, 128)) / 255.
        if classifier.weights == "imagenet":
            tiles = (tiles - tf.constant([.485, .456, .406])) / tf.constant([.229, .224, .225])
        tile_maps, probabilities = cam(tf.cast(tiles, tf.float32))
        tile_maps = tile_maps.numpy().astype(np.float32)
        native_shape = list(tile_maps.shape[1:])
        maps.append(stitch_tile_maps(tile_maps))
        classifier_scores.append(probabilities.numpy()[:, 0])
    return np.stack(maps), np.asarray(classifier_scores, dtype=np.float32), native_shape


def plot_results(result_dir, rows, saved, test_images, test_layouts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = tuple(method for method in METHODS if any(row["method"] == method for row in rows))
    labels = {"polygarbor": "PolyGabor", "resnet18": "ResNet-18 random", "resnet18_imagenet": "ResNet-18 ImageNet"}
    colors = {"polygarbor": "#1b9e77", "resnet18": "#d95f02", "resnet18_imagenet": "#e41a1c"}
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    chart_metrics = (
        ("test_patch_auc", "AUROC da explicação por patch"),
        ("test_patch_tumor_recall_at_2", "Tumores recuperados no top-2"),
        ("test_classifier_patch_auc", "AUROC da classificação por patch"),
    )
    for ax, (metric, title) in zip(axes, chart_metrics):
        for x, method in enumerate(methods):
            values = [r[metric] for r in rows if r["method"] == method]
            ax.scatter([x] * len(values), values, color=colors[method], alpha=.45, s=45)
            ax.scatter(x, np.mean(values), color=colors[method], marker="D", edgecolor="white", s=80, zorder=3)
            ax.annotate(f"{np.mean(values):.3f}", (x, np.mean(values)), xytext=(7, 0), textcoords="offset points", va="center")
        ax.set(title=title, xticks=range(len(methods)), xticklabels=[labels[m] for m in methods], ylim=(0, 1))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Identificação dos patches de tumor em 30 mosaicos de teste")
    fig.tight_layout()
    fig.savefig(result_dir / "localization_metrics.png", dpi=180)
    plt.close(fig)

    class_names = json.loads((ROOT / "dataset_manifest.json").read_text())["class_names"]

    def draw_grid(ax):
        for edge in (TILE_SIZE, 2 * TILE_SIZE, 3 * TILE_SIZE):
            ax.axhline(edge, color="white", linewidth=.7)
            ax.axvline(edge, color="white", linewidth=.7)

    def draw_truth(ax, layout, show_labels=False):
        for position, label in enumerate(layout["labels"]):
            row, col = divmod(position, GRID_SIZE)
            x, y = col * TILE_SIZE, row * TILE_SIZE
            if label == 0:
                ax.add_patch(Rectangle((x + 2, y + 2), TILE_SIZE - 4, TILE_SIZE - 4, fill=False, edgecolor="#39ff14", linewidth=2.5))
                ax.text(x + 6, y + 18, "TUMOR", color="#39ff14", fontsize=7, weight="bold", bbox=dict(facecolor="black", alpha=.72, pad=1, edgecolor="none"))
            elif show_labels:
                ax.text(x + 6, y + 18, class_names[label], color="white", fontsize=6.5, bbox=dict(facecolor="black", alpha=.62, pad=1, edgecolor="none"))

    def evidence_by_patch(score):
        score = np.asarray(score)
        return patch_scores_from_map(score) if score.ndim == 2 else score

    def draw_top2(ax, score):
        for rank, position in enumerate(np.argsort(-evidence_by_patch(score), kind="stable")[:2], 1):
            row, col = divmod(int(position), GRID_SIZE)
            x, y = col * TILE_SIZE, row * TILE_SIZE
            ax.add_patch(Rectangle((x + 6, y + 6), TILE_SIZE - 12, TILE_SIZE - 12, fill=False, edgecolor="#ffd92f", linewidth=2, linestyle="--"))
            ax.text(x + TILE_SIZE - 8, y + 18, f"P{rank}", ha="right", color="#ffd92f", fontsize=7, weight="bold", bbox=dict(facecolor="black", alpha=.72, pad=1, edgecolor="none"))

    def classification_map(scores):
        return np.repeat(np.repeat(np.asarray(scores).reshape(GRID_SIZE, GRID_SIZE), TILE_SIZE, axis=0), TILE_SIZE, axis=1)

    def draw_scores(ax, scores):
        for position, value in enumerate(scores):
            row, col = divmod(position, GRID_SIZE)
            x, y = col * TILE_SIZE, row * TILE_SIZE
            ax.text(x + TILE_SIZE / 2, y + TILE_SIZE - 8, f"{value:.2f}", ha="center", color="white", fontsize=6.5, weight="bold", bbox=dict(facecolor="black", alpha=.65, pad=1, edgecolor="none"))

    def contour_patchwise(ax, score, threshold):
        for position in range(GRID_SIZE * GRID_SIZE):
            row, col = divmod(position, GRID_SIZE)
            y, x = row * TILE_SIZE, col * TILE_SIZE
            local = score[y:y + TILE_SIZE, x:x + TILE_SIZE]
            if local.min() < threshold <= local.max():
                ax.contour(np.arange(x, x + TILE_SIZE), np.arange(y, y + TILE_SIZE), local >= threshold, levels=[.5], colors=["cyan"], linewidths=.8)

    seed_rows = {r["method"]: r for r in rows if r["model_seed"] == 42}

    def plot_examples(panel_methods, filename):
        fig, axes = plt.subplots(3, 1 + 2 * len(panel_methods), figsize=(18, 11))
        for row, index in enumerate((0, 1, 2)):
            mosaic, _ = build_mosaic(test_images, test_layouts[index])
            axes[row, 0].imshow(mosaic)
            draw_grid(axes[row, 0])
            draw_truth(axes[row, 0], test_layouts[index], show_labels=True)
            axes[row, 0].set_title(f"Mosaico {index}: rótulos conhecidos\nverde = tumor")
            for method_index, method in enumerate(panel_methods):
                classifier_col = 1 + method_index * 2
                explanation_col = classifier_col + 1
                classifier_scores = saved[(method, 42)]["test_classifier_scores"][index]
                truth = np.asarray(test_layouts[index]["labels"]) == 0
                classifier_auc = roc_auc_score(truth, classifier_scores)
                classifier_recall = truth[np.argsort(-classifier_scores)[:2]].sum() / 2
                axes[row, classifier_col].imshow(mosaic)
                axes[row, classifier_col].imshow(classification_map(classifier_scores), cmap="magma", alpha=.6, vmin=0, vmax=1)
                draw_grid(axes[row, classifier_col])
                draw_truth(axes[row, classifier_col], test_layouts[index])
                draw_top2(axes[row, classifier_col], classifier_scores)
                draw_scores(axes[row, classifier_col], classifier_scores)
                axes[row, classifier_col].set_title(f"{labels[method]} · classificação\nAUC={classifier_auc:.3f}; top-2={classifier_recall:.0%}", fontsize=9)
                score = saved[(method, 42)]["test"][index]
                threshold = seed_rows[method]["threshold"]
                axes[row, explanation_col].imshow(mosaic)
                axes[row, explanation_col].imshow(score, cmap="magma", alpha=.55, vmin=np.percentile(saved[(method, 42)]["val"], 1), vmax=np.percentile(saved[(method, 42)]["val"], 99))
                contour_patchwise(axes[row, explanation_col], score, threshold)
                draw_grid(axes[row, explanation_col])
                draw_truth(axes[row, explanation_col], test_layouts[index])
                draw_top2(axes[row, explanation_col], score)
                patch_score = patch_scores_from_map(score)
                auc = roc_auc_score(truth, patch_score)
                recall = truth[np.argsort(-patch_score)[:2]].sum() / 2
                axes[row, explanation_col].set_title(f"{labels[method]} · explicação\nAUC={auc:.3f}; top-2={recall:.0%}", fontsize=9)
            for ax in axes[row]:
                ax.axis("off")
        fig.suptitle("Classificação versus explicação · verde: tumor · amarelo: top-2 · ciano: limiar explicativo", fontsize=11)
        fig.subplots_adjust(left=.015, right=.995, bottom=.015, top=.94, wspace=.08, hspace=.2)
        fig.savefig(result_dir / filename, dpi=160)
        plt.close(fig)

    plot_examples(("polygarbor", "resnet18"), "localization_examples.png")
    if "resnet18_imagenet" in methods:
        plot_examples(("resnet18", "resnet18_imagenet"), "localization_pretraining_examples.png")


def write_report(result_dir, rows, comparisons, run_dir):
    headers = ["method", "model_seed", "test_patch_auc", "test_patch_average_precision", "test_patch_dice", "test_patch_iou", "test_patch_tumor_recall_at_2", "test_patch_both_tumors_top2_rate", "test_classifier_patch_auc", "test_classifier_patch_average_precision", "test_classifier_tumor_recall_at_2", "test_classifier_both_tumors_top2_rate", "test_pooled_auc", "test_pooled_average_precision", "test_pooled_dice", "test_pooled_iou", "patch_threshold", "threshold", "map_seconds"]
    with (result_dir / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in headers} for r in rows])
    with (result_dir / "paired_comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparisons[0])
        writer.writeheader()
        writer.writerows(comparisons)
    table = ["| Método | Seed | AUROC explicação/patch | AP explicação/patch | Recall top-2 | Ambos no top-2 | AUROC classificador/patch | AUROC região fraca | Dice região fraca |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        table.append(f"| {r['method']} | {r['model_seed']} | {r['test_patch_auc']:.4f} | {r['test_patch_average_precision']:.4f} | {r['test_patch_tumor_recall_at_2']:.4f} | {r['test_patch_both_tumors_top2_rate']:.4f} | {r['test_classifier_patch_auc']:.4f} | {r['test_pooled_auc']:.4f} | {r['test_pooled_dice']:.4f} |")
    pretraining_section = ["![Comparação de CAM por inicialização](localization_pretraining_examples.png)", "", "A comparação de inicialização mantém arquitetura, imagens e layouts e troca os pesos iniciais da ResNet. Ela permite observar separadamente alterações no ranking classificatório e no CAM."] if any(r["method"] == "resnet18_imagenet" for r in rows) else []
    text = ["# Localização quantitativa de tumor em mosaicos", "", "Foram avaliados 20 mosaicos de validação e 30 de teste, todos 4×4 e 600×600 pixels, com dois patches de tumor e quatorze não tumorais. Os 800 patches utilizados são únicos dentro de cada split. Os modelos completos das seeds 42 e 43 foram reutilizados sem retreino.", "", "![Métricas de localização](localization_metrics.png)", "", *table, "", "A análise principal usa o rótulo conhecido de cada patch. AUROC e AP verificam se a evidência média do mapa ordena patches tumorais acima dos demais. Recall top-2 mede quantos dos dois tumores aparecem entre os dois patches de maior evidência; ambos no top-2 exige acerto perfeito do par. AUROC do classificador usa seu escore de tumor para cada patch isolado e permite distinguir erro da decisão e erro do mapa explicativo.", "", "![Exemplos de mosaicos, rótulos e mapas](localization_examples.png)", "", "Cada patch é processado isoladamente e somente então os mapas são remontados. Assim, nenhum campo receptivo nem interpolação cruza as bordas artificiais do mosaico. A figura mostra uma coluna de verdade e, para cada método, uma coluna com o escore classificatório de tumor e outra com o mapa explicativo. Verde identifica a verdade tumor, amarelo tracejado mostra o top-2 de cada painel e ciano mostra o limiar da explicação selecionado na validação. Os valores PolyGabor são similaridades heurísticas e os valores ResNet são softmax não calibrado; servem para ranking dentro do método.", "", *pretraining_section, "", "O escore PolyGabor é a distância logarítmica negativa para tumor em uma grade densa 75×75 por patch. O CAM da ResNet é calculado em sua entrada treinada de 128×128, antes do softmax, a partir das ativações espaciais 4×4 e dos pesos da classe tumor, com ReLU. Cada mapa é interpolado apenas dentro do respectivo patch de 150×150.", "", "O threshold de cada método e seed maximiza Dice exclusivamente na validação. As métricas em pixels foram mantidas como análise secundária de região fracamente anotada: toda a área de um patch tumor é positiva porque não há contorno histopatológico dentro dele. Elas não medem segmentação celular ou tumoral real. O baseline aleatório de AUROC é 0,5 e a prevalência positiva é 12,5%. Os intervalos reamostram os 30 mosaicos inteiros por 5.000 draws.", "", f"Artefatos brutos: {run_dir.relative_to(ROOT)}. Tempos e recursos estão em timings.json e resources.json. A parte qualitativa nas imagens grandes não foi executada porque o arquivo local contém os 5.000 patches, sem o dataset separado colorectal_histology_large."]
    (result_dir / "README.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.campaign / "localization"
    result_dir = ROOT / "results" / args.campaign / "localization"
    status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
    if status.get("status") == "complete" and status.get("protocol_version") == PROTOCOL_VERSION:
        print(f"SKIP completed {run_dir}")
        return
    previous_rows_list = json.loads((run_dir / "raw_metrics.json").read_text()) if (run_dir / "raw_metrics.json").exists() else []
    previous_rows = {(row["method"], row["model_seed"]): row for row in previous_rows_list}
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if status.get("protocol_version") == 2:
        for name in ("status.json", "raw_metrics.json", "timings.json", "resources.json", "resources.csv"):
            source = run_dir / name
            target = run_dir / f"{source.stem}_v2{source.suffix}"
            if source.exists() and not target.exists():
                shutil.copy2(source, target)
    (run_dir / "status.json").write_text(json.dumps({"status": "running", "protocol_version": PROTOCOL_VERSION}))
    val_images = np.load(ROOT / "cache/val_images.npy", mmap_mode="r")
    val_labels = np.load(ROOT / "cache/val_labels.npy")
    test_images = np.load(ROOT / "cache/test_images.npy", mmap_mode="r")
    test_labels = np.load(ROOT / "cache/test_labels.npy")
    val_layouts = make_layouts(val_labels, 20, seed=20260912)
    test_layouts = make_layouts(test_labels, 30, seed=20260913)
    protocol = dict(protocol_version=PROTOCOL_VERSION, methods=list(METHODS), grid=[4, 4], patch_pixels=[150, 150], mosaic_pixels=[600, 600], tumors_per_mosaic=2,
                    validation_mosaics=20, test_mosaics=30, unique_within_split=True, interpolation="linear",
                    threshold_selection="maximum pooled validation Dice", bootstrap_draws=5000,
                    primary_evaluation_unit="patch", map_construction="independent explanation per patch, resized within tile, then stitched",
                    validation_layouts=val_layouts, test_layouts=test_layouts)
    (run_dir / "layout_manifest.json").write_text(json.dumps(protocol, indent=2))
    from resources import Monitor
    monitor = Monitor(run_dir)
    saved, rows = {}, []
    try:
        for method in METHODS:
            for model_seed in (42, 43):
                print(f"LOCALIZATION {method} seed={model_seed}", flush=True)
                model_dir = ROOT / "runs" / args.campaign / f"full__{method}__seed{model_seed}" / "model"
                if not model_dir.exists():
                    raise FileNotFoundError(model_dir)
                cache_path = run_dir / f"{method}_seed{model_seed}_maps.npz"
                prior = previous_rows.get((method, model_seed))
                reused = False
                if prior and cache_path.exists():
                    cached = np.load(cache_path)
                    required = {"validation", "test", "validation_classifier_scores", "test_classifier_scores"}
                    if required <= set(cached.files):
                        val_maps, test_maps = cached["validation"], cached["test"]
                        val_classifier_scores, test_classifier_scores = cached["validation_classifier_scores"], cached["test_classifier_scores"]
                        native_shape = prior["per_patch_native_shape"]
                        map_seconds = prior["map_seconds"]
                        reused = True
                        print(f"REUSE {method} seed={model_seed}", flush=True)
                if not reused:
                    started = time.perf_counter()
                    with monitor.stage(f"{method}_seed{model_seed}_maps"):
                        if method == "polygarbor":
                            val_maps, val_classifier_scores, native_shape = polygarbor_maps(model_dir, val_images, val_layouts)
                            test_maps, test_classifier_scores, check_shape = polygarbor_maps(model_dir, test_images, test_layouts)
                        else:
                            val_maps, val_classifier_scores, native_shape = resnet_maps(model_dir, val_images, val_layouts)
                            test_maps, test_classifier_scores, check_shape = resnet_maps(model_dir, test_images, test_layouts)
                        if native_shape != check_shape:
                            raise RuntimeError(f"inconsistent native map shapes: {native_shape} != {check_shape}")
                    map_seconds = time.perf_counter() - started
                with monitor.stage(f"{method}_seed{model_seed}_metrics"):
                    metrics = evaluate_maps(val_maps, test_maps, val_layouts, test_layouts, val_classifier_scores, test_classifier_scores)
                row = dict(method=method, model_seed=model_seed, map_seconds=map_seconds,
                           per_patch_native_shape=native_shape, stitched_shape=list(val_maps.shape[1:]), **metrics)
                rows.append(row)
                saved[(method, model_seed)] = {"val": val_maps, "test": test_maps, "val_classifier_scores": val_classifier_scores, "test_classifier_scores": test_classifier_scores}
                np.savez_compressed(run_dir / f"{method}_seed{model_seed}_maps.npz", validation=val_maps, test=test_maps, validation_classifier_scores=val_classifier_scores, test_classifier_scores=test_classifier_scores)
                (run_dir / "raw_metrics.json").write_text(json.dumps(rows, indent=2))
                print(f"DONE {method} seed={model_seed}: patch_AUROC={metrics['test_patch_auc']:.4f} top2_recall={metrics['test_patch_tumor_recall_at_2']:.4f}", flush=True)
        comparisons = []
        rng = np.random.default_rng(321)
        for seed in (42, 43):
            for method_a, method_b in itertools.combinations(METHODS, 2):
                row_a = next(r for r in rows if r["method"] == method_a and r["model_seed"] == seed)
                row_b = next(r for r in rows if r["method"] == method_b and r["model_seed"] == seed)
                for metric, key in (("weak_region_AUROC", "per_mosaic_auc"), ("weak_region_Dice", "per_mosaic_dice"), ("patch_AUROC", "per_mosaic_patch_auc"), ("patch_Dice", "per_mosaic_patch_dice"), ("patch_recall_at_2", "per_mosaic_patch_recall_at_2")):
                    delta = np.asarray(row_a[key]) - np.asarray(row_b[key])
                    boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(1)
                    comparisons.append(dict(model_seed=seed, method_a=method_a, method_b=method_b, metric=metric, difference_a_minus_b=float(delta.mean()), ci95_low=float(np.percentile(boot, 2.5)), ci95_high=float(np.percentile(boot, 97.5))))
        plot_results(result_dir, rows, saved, test_images, test_layouts)
        write_report(result_dir, rows, comparisons, run_dir)
        (run_dir / "status.json").write_text(json.dumps({"status": "complete", "protocol_version": PROTOCOL_VERSION}, indent=2))
    except Exception as exc:
        (run_dir / "status.json").write_text(json.dumps({"status": "failed", "protocol_version": PROTOCOL_VERSION, "error": repr(exc)}, indent=2))
        raise
    finally:
        monitor.finish()


if __name__ == "__main__":
    main()
