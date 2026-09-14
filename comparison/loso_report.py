"""Leave-one-source-out tables and figures, rebuilt from saved predictions only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loso import MANIFEST, global_labels
from report import COLORS, markdown_table, read

ROOT = Path(__file__).resolve().parent
METHODS = ("polygarbor", "resnet18", "resnet18_imagenet", "yolo11n_random", "yolo11n")
LABELS = {"polygarbor": "PolyGabor", "resnet18": "ResNet-18 · aleatória", "resnet18_imagenet": "ResNet-18 · ImageNet",
          "yolo11n_random": "YOLO11n · aleatória", "yolo11n": "YOLO11n · ImageNet"}
DRAWS = 5000


def confusion(y, p, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, p), 1)
    return cm


def f1_per_class(cm):
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1_present(cm):
    """Macro-F1 over classes with test support; a held-out source rarely contains all eight classes."""
    return float(f1_per_class(cm)[cm.sum(1) > 0].mean())


def pooled_metrics(y, p, names):
    n = len(names)
    cm = confusion(y, p, n)
    support = cm.sum(1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(n), where=support > 0)
    empty = names.index("empty")
    keep = y != empty
    others = [c for c in range(n) if c != empty]
    gmean = None if np.any(support == 0) else float(0. if np.any(recall == 0) else np.exp(np.log(recall).mean()))
    return dict(n=len(y), accuracy=float(np.mean(y == p)), macro_f1=float(f1_per_class(cm).mean()),
                balanced_accuracy=float(recall[support > 0].mean()), multiclass_gmean=gmean,
                macro_f1_without_empty=float(f1_per_class(confusion(y[keep], p[keep], n))[others].mean())), recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    parser.add_argument("--include-pilots", action="store_true", help="only to smoke-test this script on a pilot campaign")
    args = parser.parse_args()
    runs_root = ROOT / "runs" / args.campaign
    out = ROOT / "results" / args.campaign / "loso"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    names, folds = manifest["class_names"], manifest["folds"]
    n = len(names)
    labels = global_labels()
    sources = sorted(fold["test_source"] for fold in folds.values())

    fold_rows, parts = [], {}
    for folder in sorted(runs_root.glob("loso_*__*__seed*")):
        config, status = read(folder / "config.json", {}), read(folder / "status.json", {})
        if status.get("status") != "complete" or (config.get("pilot") and not args.include_pilots):
            continue
        fold = folds[config["scenario"]]
        saved = np.load(folder / "eval/test_clean/predictions.npz")
        y, p = saved["y_true"].astype(int), saved["y_pred"].astype(int)
        if not np.array_equal(y, labels[fold["test"]]):
            raise RuntimeError(f"{folder.name}: test labels differ from {MANIFEST.name}")
        cm = confusion(y, p, n)
        support = cm.sum(1)
        fold_rows.append(dict(method=config["method"], seed=config["seed"], fold=config["scenario"], test_source=fold["test_source"],
                              n=len(y), classes_present=int((support > 0).sum()), accuracy=float(np.mean(y == p)),
                              macro_f1_present=macro_f1_present(cm),
                              balanced_accuracy=float((np.diag(cm)[support > 0] / support[support > 0]).mean())))
        parts.setdefault((config["method"], config["seed"]), {})[fold["test_source"]] = (y, p)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(out / "fold_metrics.csv", index=False)
    complete = {key: value for key, value in parts.items() if sorted(value) == sources}

    split_rows = [dict(fold=name, test_source=fold["test_source"], n_train=sum(fold["counts"]["train"]),
                       n_val=sum(fold["counts"]["val"]), n_test=sum(fold["counts"]["test"]),
                       test_classes=", ".join(c for c, k in zip(names, fold["counts"]["test"]) if k),
                       train_empty=fold["counts"]["train"][names.index("empty")]) for name, fold in folds.items()]
    split_frame = pd.DataFrame(split_rows)
    split_frame.to_csv(out / "folds.csv", index=False)

    pooled_rows, recall_rows = [], []
    for (method, seed), by_source in sorted(complete.items()):
        y = np.concatenate([by_source[s][0] for s in sources])
        p = np.concatenate([by_source[s][1] for s in sources])
        metrics, recall = pooled_metrics(y, p, names)
        random_split = read(runs_root / f"full__{method}__seed{seed}" / "summary.json", {}).get("test_clean", {})
        pooled_rows.append(dict(method=method, seed=seed, **metrics, random_split_macro_f1=random_split.get("macro_f1")))
        recall_rows.append(dict(method=method, seed=seed, **dict(zip(names, recall.tolist()))))
    pooled_frame = pd.DataFrame(pooled_rows)
    recall_frame = pd.DataFrame(recall_rows)
    pooled_frame.to_csv(out / "pooled_metrics.csv", index=False)
    recall_frame.to_csv(out / "pooled_recall_per_class.csv", index=False)

    paired_rows = []
    rng = np.random.default_rng(123)
    draws = rng.integers(0, len(sources), size=(DRAWS, len(sources)))
    for seed in sorted({seed for _, seed in complete}):
        if ("polygarbor", seed) not in complete:
            continue
        base = {s: confusion(*complete[("polygarbor", seed)][s], n) for s in sources}
        for other in METHODS[1:]:
            if (other, seed) not in complete:
                continue
            for s in sources:
                if not np.array_equal(complete[("polygarbor", seed)][s][0], complete[(other, seed)][s][0]):
                    raise RuntimeError(f"Unpaired test labels for source {s}")
            rival = {s: confusion(*complete[(other, seed)][s], n) for s in sources}
            stack_a, stack_b = np.stack([base[s] for s in sources]), np.stack([rival[s] for s in sources])
            observed = macro_f1_present(stack_b.sum(0)) - macro_f1_present(stack_a.sum(0))
            boot = np.array([macro_f1_present(stack_b[d].sum(0)) - macro_f1_present(stack_a[d].sum(0)) for d in draws])
            fold_a = {s: macro_f1_present(base[s]) for s in sources}
            fold_b = {s: macro_f1_present(rival[s]) for s in sources}
            paired_rows.append(dict(seed=seed, a="polygarbor", b=other, macro_f1_b_minus_a=observed,
                                    source_bootstrap_ci_low=float(np.quantile(boot, .025)),
                                    source_bootstrap_ci_high=float(np.quantile(boot, .975)),
                                    folds_a_better=sum(fold_a[s] > fold_b[s] for s in sources),
                                    folds_b_better=sum(fold_b[s] > fold_a[s] for s in sources)))
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(out / "paired_polygarbor_vs_cnn.csv", index=False)

    methods = [m for m in METHODS if m in set(fold_frame.get("method", []))]
    if methods:
        per_fold = fold_frame.groupby(["method", "test_source"]).macro_f1_present.mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for row, method in enumerate(methods):
            color = COLORS[method]
            values = per_fold[per_fold.method == method]
            ax.scatter(values.macro_f1_present, row + np.linspace(-.12, .12, len(values)), s=36, color=color, alpha=.55,
                       edgecolor="white", linewidth=.8)
            pooled = pooled_frame[pooled_frame.method == method] if not pooled_frame.empty else pooled_frame
            if not pooled.empty:
                ax.scatter(pooled.macro_f1.mean(), row, marker="D", s=70, color=color, edgecolor="white", linewidth=1, zorder=3)
                ax.annotate(f"  {pooled.macro_f1.mean():.3f}", (pooled.macro_f1.mean(), row), va="center", fontsize=9)
                reference = pooled.random_split_macro_f1.dropna()
                if not reference.empty:
                    ax.scatter(reference.mean(), row, marker="o", s=70, facecolor="none", edgecolor="#555555", linewidth=1.4, zorder=3)
        ax.set_yticks(range(len(methods)), labels=[LABELS[m] for m in methods])
        ax.invert_yaxis()
        ax.set(xlim=(0, 1.05), xlabel="Macro-F1 (por fold: somente classes presentes na origem testada)",
               title="Desempenho em origens nunca vistas no treino")
        ax.grid(axis="x", alpha=.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        from matplotlib.lines import Line2D
        neutral = "#7a7a7a"
        ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=neutral, alpha=.55, label="Fold (origem testada)"),
                           Line2D([], [], marker="D", linestyle="", color=neutral, label="Predições dos 10 folds concatenadas"),
                           Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#555555", label="Split aleatório original")],
                  fontsize=8, loc="lower left", frameon=False)
        fig.text(.5, .01, "Pontos: média das seeds em cada uma das dez origens. Losango: macro-F1 das 5.000 predições concatenadas.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(out / "loso_macro_f1.png", dpi=180)
        plt.close(fig)
    if not recall_frame.empty:
        matrix = recall_frame.groupby("method")[names].mean().reindex([m for m in METHODS if m in set(recall_frame.method)])
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iat[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color="white" if value > .6 else "#1f1f1f")
        ax.set_xticks(range(len(names)), labels=names)
        ax.set_yticks(range(len(matrix)), labels=[LABELS[m] for m in matrix.index])
        ax.set_title("Recall por classe nas predições concatenadas (média das seeds)")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "loso_recall_per_class.png", dpi=180)
        plt.close(fig)

    summary = pd.DataFrame()
    if not pooled_frame.empty:
        grouped = pooled_frame.groupby("method")
        summary = pd.DataFrame(dict(seeds=grouped.seed.count(), macro_f1=grouped.macro_f1.mean(),
                                    macro_f1_min=grouped.macro_f1.min(), macro_f1_max=grouped.macro_f1.max(),
                                    multiclass_gmean=grouped.multiclass_gmean.mean(), balanced_accuracy=grouped.balanced_accuracy.mean(),
                                    macro_f1_without_empty=grouped.macro_f1_without_empty.mean(),
                                    random_split_macro_f1=grouped.random_split_macro_f1.mean()))
        summary["drop_vs_random_split"] = summary.random_split_macro_f1 - summary.macro_f1
        summary = summary.reindex([m for m in METHODS if m in summary.index]).reset_index()
    fold_table = (fold_frame.groupby(["test_source", "method"]).macro_f1_present.mean().unstack("method")
                  .reindex(columns=[m for m in METHODS if m in set(fold_frame.method)]).reset_index()) if not fold_frame.empty else fold_frame
    expected = len(folds) * len(METHODS)
    status_line = f"Foram encontradas {len(fold_frame)} execuções LOSO concluídas; {len(complete)} combinações método/seed têm os dez folds."
    sections = ["# Generalização entre origens (leave-one-source-out)", "",
        f"Campanha `{args.campaign}`. Arquivo gerado por `comparison/loso_report.py` a partir das predições salvas. {status_line} Cada seed completa tem {expected} execuções.", "",
        "As 5.000 imagens pertencem a dez códigos de origem `CRC-Prim-HE-01` a `CRC-Prim-HE-10`, recuperados por SHA-256 dos pixels em `source_manifest.json`. Cada fold reserva uma origem inteira para teste. A validação usa 10% de cada classe das nove origens restantes, e o treino usa o resto. Os índices foram fixados em `loso_manifest.json` antes de qualquer treino e são idênticos para todos os métodos e seeds. Hiperparâmetros, critérios de parada e teto de 350 vetores/classe do PolyGabor são os da campanha principal.", "",
        "Métrica primária: macro-F1 sobre as 5.000 predições concatenadas dos dez folds. Cada recorte é predito exatamente uma vez, por um modelo que não viu nenhum recorte da sua origem. O macro-F1 por fold considera apenas as classes presentes na origem testada, pois a maioria das origens não contém as oito classes.", "",
        "## Folds", "", markdown_table(split_frame), "",
        "A classe `empty` existe somente nas origens 06 (590 recortes) e 10 (35). Quando 06 é testada, o treino tem apenas os exemplos de `empty` da origem 10, menos os reservados para validação. Por isso também é apresentado o macro-F1 sem os recortes `empty` verdadeiros.", "",
        "## Resultado agregado", "", "![Macro-F1 por fold](loso_macro_f1.png)", "", markdown_table(summary), "",
        "`macro_f1_min`/`macro_f1_max` são os extremos entre seeds, não intervalo de confiança. `random_split_macro_f1` é o teste limpo do split aleatório original com as mesmas seeds; a queda estima quanto do desempenho original dependia de ver recortes das mesmas origens no treino.", "",
        "## Recall por classe", "", "![Recall por classe](loso_recall_per_class.png)", "",
        markdown_table(recall_frame.groupby("method")[names].mean().reindex([m for m in METHODS if m in set(recall_frame.method)]).reset_index()
                       if not recall_frame.empty else recall_frame), "",
        "## Macro-F1 por origem testada", "", markdown_table(fold_table), "",
        "## PolyGabor versus CNNs", "", markdown_table(paired_frame), "",
        "Diferença = macro-F1 concatenado de B menos PolyGabor. O intervalo reamostra as dez origens com reposição (5.000 sorteios), mantendo juntos todos os recortes de cada origem; com dez grupos ele é aproximado. As colunas de folds contam em quantas origens cada método teve maior macro-F1 nas classes presentes.", "",
        "## Limites", "",
        "Os dados vêm de dez imagens de um único instituto e de um único scanner. Os códigos de origem não são confirmados como pacientes distintos pelos metadados. O resultado mede generalização entre imagens desse acervo, não validação externa nem separação comprovada por paciente. A validação vem das mesmas origens do treino; o teste permanece disjunto por origem.", "",
        "Estes folds substituem `source_a` e `source_b` como análise de origem. Aqueles cenários usavam duas partições escolhidas manualmente com uma seed, validação sem `adipose` e `empty`, e em `source_b` 42% do teste era `empty` com 35 exemplos dessa classe no treino. Eles permanecem nos artefatos apenas como registro histórico.", ""]
    (out / "README.md").write_text("\n".join(sections))
    print(f"Wrote {out}; {len(fold_frame)} fold runs, {len(complete)} complete method/seed pairs")


if __name__ == "__main__":
    main()
