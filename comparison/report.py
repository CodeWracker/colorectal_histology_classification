"""Rebuild tables, plots and paired statistics from saved results only."""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cnn/src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from cnn.evaluation import summarize

COLORS = {"polygarbor": "#1b9e77", "resnet18": "#d95f02", "yolo11n": "#7570b3", "polygarbor_p3": "#66a61e", "polygarbor_p5": "#e6ab02", "polygarbor_aug": "#e7298a", "polygarbor_p3_aug": "#a6761d"}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def markdown_table(frame):
    if frame.empty:
        return "Ainda sem resultados."
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

    # Two seeds give a range, not a stable estimate of population variance.
    curve = test[test.scenario.str.startswith("few") | (test.scenario == "full")].copy()
    curve["examples_per_class"] = curve.scenario.map(lambda x: int(x[3:]) if x.startswith("few") else 500)
    curve.to_csv(out / "learning_curve.csv", index=False)
    curve_panels = (("Modelos principais", ("polygarbor", "resnet18", "yolo11n")),
                    ("Ablações do PolyGabor", ("polygarbor", "polygarbor_aug", "polygarbor_p3", "polygarbor_p3_aug", "polygarbor_p5")))
    for metric, label, filename in (("macro_f1", "Macro-F1", "learning_curve.png"), ("multiclass_gmean", "G-mean multiclasse", "learning_curve_gmean.png")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
        for ax, (title, methods) in zip(axes, curve_panels):
            for method in methods:
                group = curve[curve.method == method]
                stats = group.groupby("examples_per_class")[metric].agg(["mean", "min", "max"])
                ax.plot(stats.index, stats["mean"], "o-", label=method, color=COLORS[method], linewidth=2)
                ax.fill_between(stats.index, stats["min"], stats["max"], alpha=.10, color=COLORS[method])
            ax.set(title=title, xscale="log", ylim=(0, 1), xlabel="Imagens originais por classe")
            ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 500], labels=["1", "2", "5", "10", "20", "50", "100", "~500"])
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc="best")
        axes[0].set_ylabel(label + " no teste (500 recortes)")
        fig.suptitle("Desempenho por quantidade de exemplos de treino")
        fig.text(.5, .01, "Linhas: média de duas seeds; faixa: mínimo–máximo. Total de treino = 8 × valor do eixo X.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
    robust = frame[(frame.scenario == "full") & frame.evaluation.str.startswith("test_")]
    pivot = robust.pivot_table(index="evaluation", columns="method", values="macro_f1")
    pivot.to_csv(out / "robustness.csv")
    if not pivot.empty:
        ax = pivot.plot.bar(figsize=(12, 5), color=[COLORS[c] for c in pivot.columns], rot=30)
        ax.set(ylabel="Macro-F1 médio entre seeds", xlabel="Condição de teste", ylim=(0, 1))
        ax.figure.tight_layout()
        ax.figure.savefig(out / "robustness.png", dpi=180)
        plt.close(ax.figure)
    if not full.empty:
        methods = [m for m in ("yolo11n", "resnet18", "polygarbor", "polygarbor_aug", "polygarbor_p3", "polygarbor_p3_aug", "polygarbor_p5") if m in set(full.method)]
        labels = {"yolo11n": "YOLO11n", "resnet18": "ResNet-18", "polygarbor": "PolyGabor", "polygarbor_aug": "PolyGabor + aug.", "polygarbor_p3": "PolyGabor · 9 regiões", "polygarbor_p3_aug": "PolyGabor · 9 reg. + aug.", "polygarbor_p5": "PolyGabor · 25 regiões"}
        columns = (("train_seconds", "Treino (s) ↓", (15, 400), [20, 50, 100, 200]),
                   ("batch1_median_ms", "Inferência batch 1 (ms) ↓", (2, 60), [2, 5, 10, 20, 50]),
                   ("model_mb", "Modelo (MB) ↓", (.5, 60), [.5, 1, 3, 10, 50]),
                   ("macro_f1", "Macro-F1 ↑", (.35, 1), [.4, .6, .8, 1]))
        fig, axes = plt.subplots(1, 4, figsize=(15, 5.5), sharey=True)
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
        fig.suptitle("Custo e desempenho no treino completo")
        fig.text(.5, .01, "Losango: média das duas seeds; pontos: cada seed; linha: mínimo–máximo.", ha="center", fontsize=9)
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
        ax = edge_pivot.plot.bar(figsize=(9, 4), rot=0, color=[COLORS[c] for c in edge_pivot.columns])
        ax.set(ylabel="Latência mediana batch 1 (ms)", xlabel="Ambiente de inferência")
        ax.figure.tight_layout()
        ax.figure.savefig(out / "edge.png", dpi=180)
        plt.close(ax.figure)

    localization_path = out / "localization/metrics.csv"
    localization = pd.read_csv(localization_path) if localization_path.exists() else pd.DataFrame()
    localization_display = localization[["method", "model_seed", "test_patch_auc", "test_patch_average_precision", "test_patch_tumor_recall_at_2", "test_patch_both_tumors_top2_rate", "test_classifier_patch_auc"]].rename(columns={"method": "método", "model_seed": "seed", "test_patch_auc": "AUROC explicação", "test_patch_average_precision": "AP explicação", "test_patch_tumor_recall_at_2": "recall top-2", "test_patch_both_tumors_top2_rate": "ambos no top-2", "test_classifier_patch_auc": "AUROC classificador"}) if not localization.empty else localization
    localization_section = [] if localization.empty else [
        "## Localização quantitativa", "", "![Identificação dos patches de tumor](localization/localization_metrics.png)", "",
        markdown_table(localization_display), "",
        "Cada mosaico 4×4 contém dois patches de tumor e quatorze das demais classes, sem repetição dentro do split. Os mapas são calculados em cada patch isolado e remontados sem mistura entre vizinhos. A avaliação primária compara a evidência explicativa com os rótulos conhecidos por patch; métricas de região em pixels são secundárias porque não há anotação interna. Protocolo, exemplos e limitações estão no [relatório de localização](localization/README.md).", ""]

    sections = ["# Comparação de classificadores de histopatologia colorretal", "",
        f"Campanha `{args.campaign}`. Este arquivo é gerado dos artefatos salvos; a avaliação visual e a interpretação detalhada estão em [DISCUSSION.md](DISCUSSION.md). Há {len(index)} runs registradas e {len(runs)} concluídas. Consultar `run_index.csv` para falhas e caminhos, `metrics.csv` para todas as métricas e `../../PROTOCOL.md` para o protocolo.", "",
        "## Teste limpo após treinamento completo", "",
        markdown_table(full[["method", "seed", "accuracy", "macro_f1", "multiclass_gmean", "balanced_accuracy", "mcc", "nll", "ece"]]), "",
        "## Custo do treinamento completo", "",
        markdown_table(full[["method", "seed", "n_train", "originals_retained", "fitted_vectors_or_images", "epochs", "train_seconds", "model_mb", "rss_peak_mb", "vram_peak_mb"]]), "",
        "MB usa 1.000.000 bytes. RAM e VRAM são picos do processo completo, incluindo avaliação e figuras; custos por etapa estão em [stage_resources.csv](stage_resources.csv) e em resources.json/timings.json de cada run. GPU-% é global do dispositivo, incluindo o desktop. CPU-% usa 100% por núcleo. O PolyGabor padrão extrai 4000 vetores, mas ajusta no máximo 2800. Nas variantes, 9/25 patches e augmentation ampliam os vetores candidatos mantendo o teto de 350 por classe; originals_retained informa quantas imagens originais distintas chegam ao ajuste. effective_training.json detalha os valores por classe. Isso altera simultaneamente escala espacial e diversidade retida, devendo ser considerado na interpretação das ablações.", "",
        "## Uso de CPU e GPU durante ajuste", "",
        markdown_table(full[["method", "seed", "training_cpu_core_percent", "fit_cpu_core_percent", "fit_gpu_global_percent", "vram_peak_mb"]]), "",
        "training_cpu_core_percent usa CPU-segundos divididos pelo tempo total das etapas de treino; 100% corresponde a um núcleo ocupado. fit_cpu_core_percent é a média amostrada somente no ajuste. GPU-% é utilização global do dispositivo, não exclusiva do processo: atividade do desktop aparece mesmo durante PolyGabor. A VRAM por PID é medida separadamente; PolyGabor não executa operações na GPU.", "",
        "## Curva de aprendizagem", "", "![Macro-F1 versus quantidade de exemplos](learning_curve.png)", "",
        "![G-mean versus quantidade de exemplos](learning_curve_gmean.png)", "",
        "[Diferenças de interpretação entre G-mean e macro-F1](../../GMEAN_VS_MACRO_F1.md). A quantidade no eixo X sempre conta imagens originais, sem inflar o orçamento com augmentations ou patches.", "",
        "A faixa representa mínimo e máximo entre duas seeds, quando disponíveis; não é intervalo de confiança. O ponto ~500 tem 488–513 exemplos disponíveis por classe, com limite efetivo de 350 vetores/classe no PolyGabor padrão. Ausência de ponto por falha de ajuste não equivale a F1 zero. A validação permanece com 500 exemplos rotulados mesmo nos cenários de pouquíssimos exemplos de treino; esta é uma curva de escassez de treino, não de orçamento total de anotação.", "",
        "## Robustez no teste", "", "![Robustez](robustness.png)", "", markdown_table(pivot.reset_index()), "",
        "As perturbações têm severidade fixa, pares de pixels idênticos entre modelos e não representam bases externas nem novos pacientes. A avaliação adicional source_a/source_b testa origens separadas, sem misturá-la a estas perturbações. Ver parâmetros em scenarios.py.", "",
        "## Agregação espacial PolyGabor", "",
        markdown_table(aggregation_frame[aggregation_frame.scenario == "full"] if not aggregation_frame.empty else aggregation_frame), "",
        "Comparação adicional sem retreino: a decisão principal usa votação entre regiões; ranking usa argmax da similaridade normalizada das distâncias médias. As curvas principais mantêm a votação fixada no protocolo. Esta análise mostra se a divergência entre regiões explica parte do resultado; não foi usada para escolher retrospectivamente a melhor regra no teste.", "",
        "## Comparações pareadas", "", markdown_table(pd.DataFrame(pairs)), "",
        "Diferença = acurácia de B menos A. Bootstrap pareado de 5000 reamostragens de recortes; McNemar exato e ajuste de Holm entre pares/seeds mostrados. Também é apresentado bootstrap pareado por dez grupos de origem, reamostrando todos os recortes de cada origem em conjunto. Esse intervalo considera dependência por origem e deve ter preferência sobre o de recortes. McNemar e seu ajuste de Holm continuam exploratórios porque a independência entre recortes não é garantida. Nenhuma dessas análises demonstra generalização clínica.", "",
        "## Calibração", "", markdown_table(pd.DataFrame(calibrated)), "",
        "Temperatura ajustada exclusivamente na validação para cada run e aplicada ao teste sem mudar a classe vencedora. Os escores originais são softmax CNN e similaridade heurística normalizada PolyGabor; a fração de votos PolyGabor não é utilizada como probabilidade. AUROC/top-2 medem ranking; NLL/Brier/ECE precisam dessa ressalva. Consulte metrics.csv para os valores anteriores à calibração.", "",
        "## Edge e inferência", "", markdown_table(edge_frame), "",
        "cpu1 restringe afinidade a um processador lógico e bibliotecas a uma thread; cpu4 usa quatro threads; gpu usa a RTX local. Cada medição carrega novamente o modelo em processo separado e aquece antes da latência. As imagens já estão em RAM: a latência inclui pré-processamento e execução, mas exclui leitura de arquivo e visualizações. Não há emulação de ARM nem limite artificial de RAM. O processo serve como aproximação de restrição computacional em x86, não como benchmark de dispositivo edge real.", "",
        "## Generalização por origem", "",
        markdown_table(test[test.scenario.str.startswith("source_")][["method", "scenario", "n_train", "accuracy", "macro_f1", "multiclass_gmean", "balanced_accuracy"]]), "",
        "As origens foram recuperadas dos nomes por SHA-256 dos pixels. source_a testa 09/10 e source_b testa 06/09; validação em 08, com demais origens no treino. A classe empty só existe em 06/10, impossibilitando sua presença simultânea em três splits disjuntos. A validação 08 não contém adipose/empty; treino e teste preservam oito classes. Os testes têm tamanhos/composições diferentes do teste aleatório, logo diferenças não isolam somente efeito de origem. A identidade de paciente por código não foi confirmada.", "",
        *localization_section,
        "## Artefatos e limites", "",
        "Os splits são os originais do TFDS (4000/500/500) com ordem determinística, hashes auditáveis e nenhuma duplicata exata entre splits. O carregamento supervisionado expõe imagem/rótulo. A auditoria recuperou dez origens dos filenames e acrescentou splits source_a/source_b com origens separadas, descritos em source_manifest.json. Não há identificação clínica de paciente nem teste em base externa. ResNet inicia do zero; YOLO usa ImageNet e política de augmentations/otimizador do Ultralytics. A comparação mede pipelines com esses recursos diferentes.", "",
        "Cada run tem config.json, selection.json, status.json, run.log, timings.json, resources.csv/json, modelo, métricas/predições por imagem e figuras. Histórico e épocas são salvos para CNN. Pilotos ficam em campanhas distintas e não entram nas tabelas. Modelos e dados grandes permanecem no disco, ignorados pelo Git.", "",
        "Fontes: [dataset TFDS](https://www.tensorflow.org/datasets/catalog/colorectal_histology), [dados originais](https://zenodo.org/records/53169), [Ultralytics classificação](https://docs.ultralytics.com/tasks/classify/), [referência de mapas de ativação](https://keras.io/examples/vision/grad_cam/).", ""]
    (out / "REPORT.md").write_text("\n".join(sections))
    print(f"Wrote {out}; {len(runs)} completed runs")


if __name__ == "__main__":
    main()
