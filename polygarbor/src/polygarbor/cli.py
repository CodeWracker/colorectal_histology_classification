"""Interface de linha de comando do polygarbor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import console
from .classifier import PolyGaborClassifier, load_image
from .gabor import GaborConfig


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Mostra os defaults e preserva a formatacao do epilogo."""

DEFAULT_DATA_DIR = "data"
DEFAULT_MODEL_DIR = "artifacts/model"

# --------------------------------------------------------------------- helpers


def _tqdm(desc: str, total: int | None = None, disabled: bool = False):
    """Barra de progresso; `disable=None` faz o tqdm se calar fora de um terminal."""
    from tqdm import tqdm

    def wrap(iterable):
        return tqdm(
            iterable, desc=desc, total=total, disable=True if disabled else None, unit="img"
        )

    return wrap


def _figure(fig, out_dir: Path | None, name: str, show: bool) -> None:
    from . import visualize

    if out_dir is not None:
        path = visualize.save_figure(fig, out_dir / name)
        console.bullet(f"figura salva: {path}")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)


def _gabor_from_args(args: argparse.Namespace) -> GaborConfig:
    return GaborConfig(
        ksize=args.gabor_ksize,
        sigma=args.gabor_sigma,
        gamma=args.gabor_gamma,
        lambdas=tuple(args.gabor_lambdas),
        thetas=tuple(np.deg2rad(args.gabor_thetas)),
    )


def _print_model_summary(clf: PolyGaborClassifier) -> None:
    console.info("classes", f"{clf.n_classes} -> {', '.join(clf.class_names)}")
    console.info("features por vetor", clf.n_features)
    console.info("filtros de Gabor", len(clf.bank))
    console.info("niveis polinomiais", clf.num_levels)
    console.info(
        "grade de patches",
        f"{clf.patches_per_row}x{clf.patches_per_row}"
        if not clf.patch_size
        else f"patch de {clf.patch_size}px",
    )
    console.info("grade densa", f"{clf.dense_grid}x{clf.dense_grid} janela {clf.dense_window}")


# ------------------------------------------------------------------- comandos


def cmd_dataset(args: argparse.Namespace) -> int:
    from . import dataset as ds_mod

    console.title("FLUXO 1 - CARGA PERMANENTE DO DATASET")
    console.info("dataset", args.name)
    console.info("pasta de dados", Path(args.data_dir).expanduser().resolve())

    with console.step("Baixando / carregando via TensorFlow Datasets"):
        bundle = ds_mod.load(args.data_dir, name=args.name, shuffle_files=args.shuffle)

    console.section("Resumo do dataset")
    console.info("classes", bundle.class_names)
    console.info("shape das imagens", bundle.image_shape)
    console.table(
        [(k, v) for k, v in bundle.num_examples.items()], headers=["split", "imagens"]
    )
    console.info("cache permanente em", bundle.data_dir)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.preview and out_dir is not None:
        from . import visualize

        if not args.show:
            visualize.use_headless()
        console.section("Gerando pre-visualizacao")
        samples = ds_mod.iter_numpy(bundle["train"], limit=args.preview)
        fig = visualize.plot_dataset_grid(samples, bundle.class_names, n=args.preview)
        _figure(fig, out_dir, "dataset_samples.png", args.show)

    if args.export_per_class:
        console.section("Exportando imagens de exemplo")
        target = (out_dir or Path("artifacts")) / "samples"
        written = ds_mod.export_samples(
            bundle, target, split=args.export_split, per_class=args.export_per_class
        )
        for path in written:
            console.bullet(str(path))
        console.ok(f"{len(written)} imagens exportadas para {target}")

    console.ok("Dataset pronto. Proximo passo: `polygarbor train`.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from . import dataset as ds_mod
    from . import evaluation, visualize

    if not args.show:
        visualize.use_headless()

    console.title("TREINO - GABOR + MAHALANOBIS POLINOMIAL")
    with console.step("Carregando dataset"):
        bundle = ds_mod.load(args.data_dir, name=args.name)
    console.info("classes", ", ".join(bundle.class_names))
    console.info("imagens", bundle.num_examples)

    clf = PolyGaborClassifier(
        class_names=bundle.class_names,
        gabor=_gabor_from_args(args),
        patch_size=args.patch_size,
        patches_per_row=args.patches_per_row,
        num_levels=args.levels,
        max_samples_per_class=args.max_samples,
        dense_grid=args.dense_grid,
        dense_window=(args.dense_window, args.dense_window),
        random_state=args.seed,
    )
    console.section("Configuracao do modelo")
    _print_model_summary(clf)
    console.info("lado do patch", f"{clf.patch_size_for(bundle.image_size)}px")

    out_dir = Path(args.out_dir)
    if args.figures:
        fig = visualize.plot_gabor_bank(clf.bank)
        _figure(fig, out_dir, "gabor_bank.png", args.show)

    n_train = args.limit or bundle.num_examples["train"]
    with console.step(f"Extraindo descritores de treino ({n_train} imagens)"):
        clf.fit(
            ds_mod.iter_numpy(bundle["train"], limit=args.limit),
            progress=_tqdm("treino", total=n_train, disabled=args.quiet),
        )
    for idx, name in enumerate(clf.class_names):
        console.bullet(f"{name}: subespaco com {len(clf.train_samples[idx])} amostras")

    with console.step("Salvando modelo"):
        path = clf.save(args.model_dir)
    console.info("modelo", path.resolve())

    if not args.skip_eval:
        split = "val"
        n_eval = args.eval_limit or bundle.num_examples[split]
        with console.step(f"Avaliando no split '{split}' ({n_eval} imagens)"):
            result = evaluation.evaluate(
                clf,
                ds_mod.iter_numpy(bundle[split], limit=args.eval_limit),
                method=args.method,
                aggregation=args.aggregation,
                progress=_tqdm(split, total=n_eval, disabled=args.quiet),
            )
        console.section(f"Metricas ({split})")
        console.info("acuracia", f"{result.accuracy:.3f}")
        print()
        print(result.report_text())
        paths = result.save(out_dir)
        for key, p in paths.items():
            console.bullet(f"{key}: {p}")
        fig = visualize.plot_confusion_matrix(
            result.confusion(), clf.class_names,
            title=f"Matriz de Confusao Normalizada ({split})",
        )
        _figure(fig, out_dir, "confusion_matrix.png", args.show)

    console.ok(f"Treino concluido. Use: polygarbor predict --model-dir {args.model_dir} -i <imagem>")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from . import dataset as ds_mod
    from . import evaluation, visualize

    if not args.show:
        visualize.use_headless()

    console.title(f"AVALIACAO - SPLIT '{args.split}'")
    clf = PolyGaborClassifier.load(args.model_dir)
    _print_model_summary(clf)

    with console.step("Carregando dataset"):
        bundle = ds_mod.load(args.data_dir, name=args.name)

    n_eval = args.limit or bundle.num_examples[args.split]
    with console.step(f"Classificando {n_eval} imagens"):
        result = evaluation.evaluate(
            clf,
            ds_mod.iter_numpy(bundle[args.split], limit=args.limit),
            method=args.method,
            aggregation=args.aggregation,
            progress=_tqdm(args.split, total=n_eval, disabled=args.quiet),
        )

    console.section("Metricas")
    console.info("acuracia", f"{result.accuracy:.3f}")
    print()
    print(result.report_text())

    out_dir = Path(args.out_dir)
    for key, p in result.save(out_dir).items():
        console.bullet(f"{key}: {p}")
    fig = visualize.plot_confusion_matrix(
        result.confusion(), clf.class_names,
        title=f"Matriz de Confusao Normalizada ({args.split})",
    )
    _figure(fig, out_dir, "confusion_matrix.png", args.show)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from . import visualize

    if not args.show:
        visualize.use_headless()

    image_path = Path(args.image)
    console.title("FLUXO 2 - CLASSIFICACAO DE UMA IMAGEM")
    console.info("imagem", image_path.resolve())
    console.info("modelo", Path(args.model_dir).resolve())

    clf = PolyGaborClassifier.load(args.model_dir)
    image = load_image(image_path)
    console.info("dimensoes", f"{image.shape[1]}x{image.shape[0]}px")
    _print_model_summary(clf)

    with console.step("Classificando"):
        pred = clf.predict(image, method=args.method, aggregation=args.aggregation)

    console.section("Resultado")
    console.info("classe predita", f"{pred.class_name} (id {pred.label})")
    console.info("confianca", f"{pred.confidence:.1%}")
    console.info("unidades avaliadas", len(pred.distances))
    if args.true_label is not None:
        hit = "ACERTO" if pred.label == args.true_label else "ERRO"
        console.info("ground truth", f"{clf.class_names[args.true_label]} -> {hit}")

    console.section("Ranking de classes")
    console.table(
        [
            (name, f"{sim:.4g}", votes, f"{dist:,.2f}")
            for name, sim, votes, dist in pred.ranking(clf.class_names)
        ],
        headers=["classe", "similaridade", "votos", "dist. media"],
    )

    out_dir = Path(args.out_dir) / image_path.stem if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = pred.to_dict(clf.class_names)
        payload["image"] = str(image_path.resolve())
        (out_dir / "prediction.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        console.bullet(f"json salvo: {out_dir / 'prediction.json'}")

    if args.no_viz:
        console.ok("Concluido (visualizacoes desativadas).")
        return 0

    console.section("Gerando visualizacoes")
    _figure(
        visualize.plot_prediction_summary(
            image, pred, clf.class_names, true_label=args.true_label
        ),
        out_dir, "01_resultado.png", args.show,
    )
    _figure(visualize.plot_gabor_bank(clf.bank), out_dir, "02_banco_gabor.png", args.show)

    feats, views = clf.extract_with_views(image)
    if views:
        idx = min(args.patch_index, len(views) - 1)
        _figure(
            visualize.plot_patch_decomposition(
                views[idx], clf.bank.labels,
                title=f"Decomposicao do patch {idx} - {pred.class_name}",
            ),
            out_dir, "03_decomposicao_patch.png", args.show,
        )
        _figure(
            visualize.plot_feature_matrix(feats, clf.feature_names),
            out_dir, "04_matriz_descritora.png", args.show,
        )

    dense_pred = (
        pred if pred.grid is not None
        else clf.predict(image, method="dense", aggregation=args.aggregation)
    )
    _figure(
        visualize.plot_similarity_maps(
            image, dense_pred, clf.class_names,
            true_label=args.true_label, gamma=clf.similarity_gamma,
        ),
        out_dir, "05_mapas_similaridade.png", args.show,
    )
    if dense_pred is not pred:
        console.bullet(
            f"voto denso (por pixel): {dense_pred.class_name} "
            f"({dense_pred.confidence:.1%})"
        )

    console.ok(f"Concluido. Saidas em {out_dir}" if out_dir else "Concluido.")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    console.title("INFORMACOES DO MODELO")
    model_dir = Path(args.model_dir)
    clf = PolyGaborClassifier.load(model_dir)
    console.info("pasta", model_dir.resolve())
    _print_model_summary(clf)
    console.section("Amostras de treino por classe")
    console.table(
        [(clf.class_names[i], len(s), s.shape[1]) for i, s in sorted(clf.train_samples.items())],
        headers=["classe", "amostras", "dimensoes"],
    )
    console.section("Features do descritor")
    console.info("ordem", ", ".join(clf.feature_names))
    return 0


# ----------------------------------------------------------------- argparse


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-q", "--quiet", action="store_true", help="oculta barras de progresso")
    parser.add_argument("--show", action="store_true", help="abre as figuras em janela")
    parser.add_argument("--no-color", action="store_true", help="desativa cores no terminal")


def _add_gabor(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("banco de Gabor")
    group.add_argument("--gabor-ksize", type=int, default=21, help="lado do kernel (default: 21)")
    group.add_argument("--gabor-sigma", type=float, default=3.0, help="sigma da gaussiana (default: 3.0)")
    group.add_argument("--gabor-gamma", type=float, default=0.5, help="razao de aspecto (default: 0.5)")
    group.add_argument("--gabor-lambdas", type=float, nargs="+", default=[4.0, 8.0],
                       help="comprimentos de onda (default: 4 8)")
    group.add_argument("--gabor-thetas", type=float, nargs="+", default=[0, 45, 90, 135],
                       help="orientacoes em graus (default: 0 45 90 135)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polygarbor",
        description=(
            "Classificacao de histologia colorretal com filtros de Gabor e "
            "Distancia de Mahalanobis Polinomial."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "exemplos:\n"
            "  polygarbor dataset --data-dir ./data --preview 9 --export-per-class 1\n"
            "  polygarbor train --data-dir ./data --model-dir ./artifacts/model\n"
            "  polygarbor predict -i lamina.png --model-dir ./artifacts/model\n"
        ),
    )
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command", metavar="comando")

    # dataset -----------------------------------------------------------
    p_ds = sub.add_parser(
        "dataset", help="baixa e materializa o dataset numa pasta permanente",
        formatter_class=_HelpFormatter,
    )
    p_ds.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="pasta permanente do TFDS")
    p_ds.add_argument("--name", default="colorectal_histology", help="nome do dataset no TFDS")
    p_ds.add_argument("--out-dir", default="artifacts/dataset", help="pasta das figuras/exports")
    p_ds.add_argument("--preview", type=int, default=9, help="quantas amostras plotar (0 desativa)")
    p_ds.add_argument("--export-per-class", type=int, default=0,
                      help="exporta N imagens PNG por classe para testar o predict")
    p_ds.add_argument("--export-split", default="test", choices=["train", "val", "test"],
                      help="split usado na exportacao")
    p_ds.add_argument("--shuffle", action="store_true",
                      help="embaralha a ordem de leitura (deixa a preview variada)")
    _add_common(p_ds)
    p_ds.set_defaults(func=cmd_dataset)

    # train -------------------------------------------------------------
    p_tr = sub.add_parser(
        "train", help="extrai descritores do dataset e ajusta os subespacos",
        formatter_class=_HelpFormatter,
    )
    p_tr.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="pasta permanente do TFDS")
    p_tr.add_argument("--name", default="colorectal_histology", help="nome do dataset no TFDS")
    p_tr.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="onde salvar o modelo")
    p_tr.add_argument("--out-dir", default="artifacts/train", help="pasta de figuras e metricas")
    p_tr.add_argument("--patches-per-row", type=int, default=1,
                      help="divide a imagem em NxN patches (1 = imagem inteira)")
    p_tr.add_argument("--patch-size", type=int, default=None,
                      help="lado do patch em px (sobrepoe --patches-per-row)")
    p_tr.add_argument("--levels", type=int, default=3, help="niveis da expansao polinomial")
    p_tr.add_argument("--max-samples", type=int, default=350,
                      help="maximo de vetores de treino por classe")
    p_tr.add_argument("--limit", type=int, default=None, help="usa apenas N imagens de treino")
    p_tr.add_argument("--eval-limit", type=int, default=None, help="usa apenas N imagens na avaliacao")
    p_tr.add_argument("--skip-eval", action="store_true", help="nao avalia apos treinar")
    p_tr.add_argument("--method", default="patch", choices=["patch", "dense"],
                      help="descritor usado na avaliacao")
    p_tr.add_argument("--aggregation", default="voting", choices=["voting", "mean"],
                      help="como combinar as unidades numa decisao")
    p_tr.add_argument("--dense-grid", type=int, default=75, help="lado da grade densa")
    p_tr.add_argument("--dense-window", type=int, default=15, help="lado da janela local densa")
    p_tr.add_argument("--figures", action="store_true", help="salva a figura do banco de Gabor")
    p_tr.add_argument("--seed", type=int, default=42, help="semente da amostragem")
    _add_gabor(p_tr)
    _add_common(p_tr)
    p_tr.set_defaults(func=cmd_train)

    # evaluate ----------------------------------------------------------
    p_ev = sub.add_parser(
        "evaluate", help="avalia um modelo salvo em um split do dataset",
        formatter_class=_HelpFormatter,
    )
    p_ev.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="modelo salvo")
    p_ev.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="pasta permanente do TFDS")
    p_ev.add_argument("--name", default="colorectal_histology", help="nome do dataset no TFDS")
    p_ev.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_ev.add_argument("--out-dir", default="artifacts/eval", help="pasta de metricas e figuras")
    p_ev.add_argument("--limit", type=int, default=None, help="usa apenas N imagens")
    p_ev.add_argument("--method", default="patch", choices=["patch", "dense"])
    p_ev.add_argument("--aggregation", default="voting", choices=["voting", "mean"])
    _add_common(p_ev)
    p_ev.set_defaults(func=cmd_evaluate)

    # predict -----------------------------------------------------------
    p_pr = sub.add_parser(
        "predict", help="classifica uma imagem e gera as visualizacoes",
        formatter_class=_HelpFormatter,
    )
    p_pr.add_argument("-i", "--image", required=True, help="caminho da imagem a classificar")
    p_pr.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="modelo salvo")
    p_pr.add_argument("--out-dir", default="artifacts/predict", help="pasta das saidas")
    p_pr.add_argument("--method", default="patch", choices=["patch", "dense"],
                      help="descritor usado na decisao final")
    p_pr.add_argument("--aggregation", default="voting", choices=["voting", "mean"])
    p_pr.add_argument("--true-label", type=int, default=None,
                      help="id da classe real, se conhecida (destaca acerto/erro)")
    p_pr.add_argument("--patch-index", type=int, default=0,
                      help="qual patch detalhar na figura de decomposicao")
    p_pr.add_argument("--no-viz", action="store_true", help="apenas classifica, sem figuras")
    _add_common(p_pr)
    p_pr.set_defaults(func=cmd_predict)

    # info --------------------------------------------------------------
    p_in = sub.add_parser("info", help="mostra a configuracao de um modelo salvo",
                          formatter_class=_HelpFormatter)
    p_in.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    _add_common(p_in)
    p_in.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.func is None:
        parser.print_help()
        return 1
    if getattr(args, "no_color", False):
        console.disable_color()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.error("interrompido pelo usuario")
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        console.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
