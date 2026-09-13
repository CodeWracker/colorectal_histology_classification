"""Command line interface for polygarbor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import console
from .classifier import PolyGaborClassifier, load_image
from .gabor import GaborConfig

DEFAULT_DATA_DIR = "data"
DEFAULT_MODEL_DIR = "artifacts/model"


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show defaults while preserving the epilog's formatting."""


# --------------------------------------------------------------------- helpers


def _tqdm(desc: str, total: int | None = None, disabled: bool = False):
    """Progress bar; `disable=None` makes tqdm stay quiet outside a terminal."""
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
        console.bullet(f"figure saved: {path}")
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
    console.info("features per vector", clf.n_features)
    console.info("gabor filters", len(clf.bank))
    console.info("polynomial levels", clf.num_levels)
    console.info(
        "patch grid",
        f"{clf.patches_per_row}x{clf.patches_per_row}"
        if not clf.patch_size
        else f"{clf.patch_size}px patch",
    )
    console.info("dense grid", f"{clf.dense_grid}x{clf.dense_grid} window {clf.dense_window}")


# -------------------------------------------------------------------- commands


def cmd_dataset(args: argparse.Namespace) -> int:
    from . import dataset as ds_mod

    console.title("FLOW 1 - PERMANENT DATASET LOADING")
    console.info("dataset", args.name)
    console.info("data folder", Path(args.data_dir).expanduser().resolve())

    with console.step("Downloading / loading through TensorFlow Datasets"):
        bundle = ds_mod.load(args.data_dir, name=args.name, shuffle_files=args.shuffle)

    console.section("Dataset summary")
    console.info("classes", bundle.class_names)
    console.info("image shape", bundle.image_shape)
    console.table(
        [(k, v) for k, v in bundle.num_examples.items()], headers=["split", "images"]
    )
    console.info("permanent cache at", bundle.data_dir)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.preview and out_dir is not None:
        from . import visualize

        if not args.show:
            visualize.use_headless()
        console.section("Generating preview")
        samples = ds_mod.iter_numpy(bundle["train"], limit=args.preview)
        fig = visualize.plot_dataset_grid(samples, bundle.class_names, n=args.preview)
        _figure(fig, out_dir, "dataset_samples.png", args.show)

    if args.export_per_class:
        console.section("Exporting example images")
        target = (out_dir or Path("artifacts")) / "samples"
        written = ds_mod.export_samples(
            bundle, target, split=args.export_split, per_class=args.export_per_class
        )
        for path in written:
            console.bullet(str(path))
        console.ok(f"{len(written)} images exported to {target}")

    console.ok(f"Dataset ready. Next step: `{getattr(args, 'program', 'polygarbor')} train`.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from . import dataset as ds_mod
    from . import evaluation, visualize

    if not args.show:
        visualize.use_headless()

    console.title("TRAINING - GABOR + POLYNOMIAL MAHALANOBIS")
    with console.step("Loading dataset"):
        bundle = ds_mod.load(args.data_dir, name=args.name)
    console.info("classes", ", ".join(bundle.class_names))
    console.info("images", bundle.num_examples)

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
    console.section("Model configuration")
    _print_model_summary(clf)
    console.info("patch side", f"{clf.patch_size_for(bundle.image_size)}px")

    out_dir = Path(args.out_dir)
    if args.figures:
        fig = visualize.plot_gabor_bank(clf.bank)
        _figure(fig, out_dir, "gabor_bank.png", args.show)

    n_train = args.limit or bundle.num_examples["train"]
    with console.step(f"Extracting training descriptors ({n_train} images)"):
        clf.fit(
            ds_mod.iter_numpy(bundle["train"], limit=args.limit),
            progress=_tqdm("train", total=n_train, disabled=args.quiet),
        )
    for idx, name in enumerate(clf.class_names):
        console.bullet(f"{name}: subspace with {len(clf.train_samples[idx])} samples")

    with console.step("Saving model"):
        path = clf.save(args.model_dir)
    console.info("model", path.resolve())

    if not args.skip_eval:
        split = "val"
        n_eval = args.eval_limit or bundle.num_examples[split]
        with console.step(f"Evaluating on the '{split}' split ({n_eval} images)"):
            result = evaluation.evaluate(
                clf,
                ds_mod.iter_numpy(bundle[split], limit=args.eval_limit),
                method=args.method,
                aggregation=args.aggregation,
                progress=_tqdm(split, total=n_eval, disabled=args.quiet),
            )
        console.section(f"Metrics ({split})")
        console.info("accuracy", f"{result.accuracy:.3f}")
        print()
        print(result.report_text())
        paths = result.save(out_dir)
        for key, p in paths.items():
            console.bullet(f"{key}: {p}")
        fig = visualize.plot_confusion_matrix(
            result.confusion(), clf.class_names,
            title=f"Normalized confusion matrix ({split})",
        )
        _figure(fig, out_dir, "confusion_matrix.png", args.show)

    console.ok(f"Training done. Try: polygarbor predict --model-dir {args.model_dir} -i <image>")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from . import dataset as ds_mod
    from . import evaluation, visualize

    if not args.show:
        visualize.use_headless()

    console.title(f"EVALUATION - '{args.split}' SPLIT")
    clf = PolyGaborClassifier.load(args.model_dir)
    _print_model_summary(clf)

    with console.step("Loading dataset"):
        bundle = ds_mod.load(args.data_dir, name=args.name)

    n_eval = args.limit or bundle.num_examples[args.split]
    with console.step(f"Classifying {n_eval} images"):
        result = evaluation.evaluate(
            clf,
            ds_mod.iter_numpy(bundle[args.split], limit=args.limit),
            method=args.method,
            aggregation=args.aggregation,
            progress=_tqdm(args.split, total=n_eval, disabled=args.quiet),
        )

    console.section("Metrics")
    console.info("accuracy", f"{result.accuracy:.3f}")
    print()
    print(result.report_text())

    out_dir = Path(args.out_dir)
    for key, p in result.save(out_dir).items():
        console.bullet(f"{key}: {p}")
    fig = visualize.plot_confusion_matrix(
        result.confusion(), clf.class_names,
        title=f"Normalized confusion matrix ({args.split})",
    )
    _figure(fig, out_dir, "confusion_matrix.png", args.show)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from . import visualize

    if not args.show:
        visualize.use_headless()

    image_path = Path(args.image)
    console.title("FLOW 2 - SINGLE IMAGE CLASSIFICATION")
    console.info("image", image_path.resolve())
    console.info("model", Path(args.model_dir).resolve())

    clf = PolyGaborClassifier.load(args.model_dir)
    image = load_image(image_path)
    console.info("dimensions", f"{image.shape[1]}x{image.shape[0]}px")
    _print_model_summary(clf)

    with console.step("Classifying"):
        pred = clf.predict(image, method=args.method, aggregation=args.aggregation)

    console.section("Result")
    console.info("predicted class", f"{pred.class_name} (id {pred.label})")
    console.info("confidence", f"{pred.confidence:.1%}")
    console.info("units evaluated", len(pred.distances))
    if args.true_label is not None:
        hit = "HIT" if pred.label == args.true_label else "MISS"
        console.info("ground truth", f"{clf.class_names[args.true_label]} -> {hit}")

    console.section("Class ranking")
    console.table(
        [
            (name, f"{sim:.4g}", votes, f"{dist:,.2f}")
            for name, sim, votes, dist in pred.ranking(clf.class_names)
        ],
        headers=["class", "similarity", "votes", "mean dist."],
    )

    out_dir = Path(args.out_dir) / image_path.stem if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = pred.to_dict(clf.class_names)
        payload["image"] = str(image_path.resolve())
        (out_dir / "prediction.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        console.bullet(f"json saved: {out_dir / 'prediction.json'}")

    if args.no_viz:
        console.ok("Done (visualizations disabled).")
        return 0

    console.section("Generating visualizations")
    _figure(
        visualize.plot_prediction_summary(
            image, pred, clf.class_names, true_label=args.true_label
        ),
        out_dir, "01_result.png", args.show,
    )
    _figure(visualize.plot_gabor_bank(clf.bank), out_dir, "02_gabor_bank.png", args.show)

    feats, views = clf.extract_with_views(image)
    if views:
        idx = min(args.patch_index, len(views) - 1)
        _figure(
            visualize.plot_patch_decomposition(
                views[idx], clf.bank.labels,
                title=f"Decomposition of patch {idx} - {pred.class_name}",
            ),
            out_dir, "03_patch_decomposition.png", args.show,
        )
        _figure(
            visualize.plot_feature_matrix(feats, clf.feature_names),
            out_dir, "04_descriptor_matrix.png", args.show,
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
        out_dir, "05_similarity_maps.png", args.show,
    )
    if dense_pred is not pred:
        console.bullet(
            f"dense (per-pixel) vote: {dense_pred.class_name} "
            f"({dense_pred.confidence:.1%})"
        )

    console.ok(f"Done. Outputs in {out_dir}" if out_dir else "Done.")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    console.title("MODEL INFORMATION")
    model_dir = Path(args.model_dir)
    clf = PolyGaborClassifier.load(model_dir)
    console.info("folder", model_dir.resolve())
    _print_model_summary(clf)
    console.section("Training samples per class")
    console.table(
        [(clf.class_names[i], len(s), s.shape[1]) for i, s in sorted(clf.train_samples.items())],
        headers=["class", "samples", "dimensions"],
    )
    console.section("Descriptor features")
    console.info("order", ", ".join(clf.feature_names))
    return 0


# ------------------------------------------------------------------- argparse


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-q", "--quiet", action="store_true", help="hide progress bars")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    parser.add_argument("--no-color", action="store_true", help="disable terminal colors")


def _add_gabor(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("gabor bank")
    group.add_argument("--gabor-ksize", type=int, default=21, help="kernel side")
    group.add_argument("--gabor-sigma", type=float, default=3.0, help="gaussian sigma")
    group.add_argument("--gabor-gamma", type=float, default=0.5, help="aspect ratio")
    group.add_argument("--gabor-lambdas", type=float, nargs="+", default=[4.0, 8.0],
                       help="wavelengths")
    group.add_argument("--gabor-thetas", type=float, nargs="+", default=[0, 45, 90, 135],
                       help="orientations in degrees")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polygarbor",
        description=(
            "Colorectal histology classification with Gabor filters and the "
            "polynomial Mahalanobis distance."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "examples:\n"
            "  polygarbor dataset --data-dir ./data --preview 9 --export-per-class 1\n"
            "  polygarbor train --data-dir ./data --model-dir ./artifacts/model\n"
            "  polygarbor predict -i slide.png --model-dir ./artifacts/model\n"
        ),
    )
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command", metavar="command")

    # dataset -----------------------------------------------------------
    p_ds = sub.add_parser(
        "dataset", help="download and materialize the dataset into a permanent folder",
        formatter_class=_HelpFormatter,
    )
    p_ds.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="permanent TFDS folder")
    p_ds.add_argument("--name", default="colorectal_histology", help="TFDS dataset name")
    p_ds.add_argument("--out-dir", default="artifacts/dataset", help="figures/exports folder")
    p_ds.add_argument("--preview", type=int, default=9, help="how many samples to plot (0 disables)")
    p_ds.add_argument("--export-per-class", type=int, default=0,
                      help="export N PNG images per class to try out predict")
    p_ds.add_argument("--export-split", default="test", choices=["train", "val", "test"],
                      help="split used for the export")
    p_ds.add_argument("--shuffle", action="store_true",
                      help="shuffle the read order (varies the preview)")
    _add_common(p_ds)
    p_ds.set_defaults(func=cmd_dataset)

    # train -------------------------------------------------------------
    p_tr = sub.add_parser(
        "train", help="extract dataset descriptors and fit the subspaces",
        formatter_class=_HelpFormatter,
    )
    p_tr.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="permanent TFDS folder")
    p_tr.add_argument("--name", default="colorectal_histology", help="TFDS dataset name")
    p_tr.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="where to save the model")
    p_tr.add_argument("--out-dir", default="artifacts/train", help="figures and metrics folder")
    p_tr.add_argument("--patches-per-row", type=int, default=1,
                      help="split the image into NxN patches (1 = whole image)")
    p_tr.add_argument("--patch-size", type=int, default=None,
                      help="patch side in px (overrides --patches-per-row)")
    p_tr.add_argument("--levels", type=int, default=3, help="polynomial expansion levels")
    p_tr.add_argument("--max-samples", type=int, default=350,
                      help="maximum training vectors per class")
    p_tr.add_argument("--limit", type=int, default=None, help="use only N training images")
    p_tr.add_argument("--eval-limit", type=int, default=None, help="use only N images for evaluation")
    p_tr.add_argument("--skip-eval", action="store_true", help="do not evaluate after training")
    p_tr.add_argument("--method", default="patch", choices=["patch", "dense"],
                      help="descriptor used for evaluation")
    p_tr.add_argument("--aggregation", default="voting", choices=["voting", "mean"],
                      help="how to combine units into one decision")
    p_tr.add_argument("--dense-grid", type=int, default=75, help="dense grid side")
    p_tr.add_argument("--dense-window", type=int, default=15, help="dense local window side")
    p_tr.add_argument("--figures", action="store_true", help="save the filter bank figure")
    p_tr.add_argument("--seed", type=int, default=42, help="sampling seed")
    _add_gabor(p_tr)
    _add_common(p_tr)
    p_tr.set_defaults(func=cmd_train)

    # evaluate ----------------------------------------------------------
    p_ev = sub.add_parser(
        "evaluate", help="evaluate a saved model on a dataset split",
        formatter_class=_HelpFormatter,
    )
    p_ev.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="saved model")
    p_ev.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="permanent TFDS folder")
    p_ev.add_argument("--name", default="colorectal_histology", help="TFDS dataset name")
    p_ev.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_ev.add_argument("--out-dir", default="artifacts/eval", help="metrics and figures folder")
    p_ev.add_argument("--limit", type=int, default=None, help="use only N images")
    p_ev.add_argument("--method", default="patch", choices=["patch", "dense"])
    p_ev.add_argument("--aggregation", default="voting", choices=["voting", "mean"])
    _add_common(p_ev)
    p_ev.set_defaults(func=cmd_evaluate)

    # predict -----------------------------------------------------------
    p_pr = sub.add_parser(
        "predict", help="classify one image and generate the visualizations",
        formatter_class=_HelpFormatter,
    )
    p_pr.add_argument("-i", "--image", required=True, help="path to the image to classify")
    p_pr.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="saved model")
    p_pr.add_argument("--out-dir", default="artifacts/predict", help="outputs folder")
    p_pr.add_argument("--method", default="patch", choices=["patch", "dense"],
                      help="descriptor used for the final decision")
    p_pr.add_argument("--aggregation", default="voting", choices=["voting", "mean"])
    p_pr.add_argument("--true-label", type=int, default=None,
                      help="id of the real class, if known (highlights hit/miss)")
    p_pr.add_argument("--patch-index", type=int, default=0,
                      help="which patch to detail in the decomposition figure")
    p_pr.add_argument("--no-viz", action="store_true", help="only classify, no figures")
    _add_common(p_pr)
    p_pr.set_defaults(func=cmd_predict)

    # info --------------------------------------------------------------
    p_in = sub.add_parser("info", help="show the configuration of a saved model",
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
        console.error("interrupted by the user")
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        console.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
