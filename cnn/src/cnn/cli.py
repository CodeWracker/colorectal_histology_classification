"""Equivalent dataset/train/evaluate/predict/info command flows."""
import argparse
import json
from pathlib import Path

from .classifier import CNNClassifier


def build_parser():
    parser = argparse.ArgumentParser(prog="cnn", description="Colorectal histology CNNs")
    sub = parser.add_subparsers(dest="command", required=True)
    # Keep the dataset command identical, including previews and exports.
    from polygarbor.cli import build_parser as poly_parser
    dataset_parser = next(a for a in poly_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices["dataset"]
    sub.add_parser("dataset", parents=[dataset_parser], add_help=False).set_defaults(program="cnn")
    for command in ("train", "evaluate", "predict", "info"):
        p = sub.add_parser(command, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument("--model-dir", default="artifacts/model")
        p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
        p.add_argument("-q", "--quiet", action="store_true")
        p.add_argument("--show", action="store_true")
        p.add_argument("--no-color", action="store_true")
        if command in ("train", "evaluate"):
            p.add_argument("--data-dir", default="data")
            p.add_argument("--name", default="colorectal_histology")
            p.add_argument("--limit", type=int)
        if command != "info":
            p.add_argument("--out-dir", default=f"artifacts/{command}")
        if command == "train":
            p.add_argument("--architecture", choices=["resnet18", "yolo11n"], default="resnet18")
            p.add_argument("--image-size", type=int, default=128)
            p.add_argument("--batch-size", type=int, default=32)
            p.add_argument("--learning-rate", type=float, default=.001)
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--patience", type=int, default=8)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--threads", type=int, default=4)
            p.add_argument("--no-augment", action="store_true")
            p.add_argument("--weights", default="auto", help="auto, random, imagenet (ResNet), or a YOLO checkpoint/config")
            p.add_argument("--eval-limit", type=int)
            p.add_argument("--skip-eval", action="store_true")
            p.add_argument("--figures", action="store_true")
        elif command == "evaluate":
            p.add_argument("--split", choices=["train", "val", "test"], default="test")
        elif command == "predict":
            p.add_argument("-i", "--image", required=True)
            p.add_argument("--true-label", type=int)
            p.add_argument("--no-viz", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_color:
        from . import console
        console.disable_color()
    try:
        if args.command == "dataset":
            return args.func(args)
        from . import dataset, visualize
        from .evaluation import save_evaluation
        from polygarbor.classifier import load_image
        import matplotlib.pyplot as plt
        if not args.show:
            visualize.use_headless()
        for key in ("limit", "eval_limit"):
            if getattr(args, key, None) is not None and getattr(args, key) < 1:
                raise ValueError(f"--{key.replace('_', '-')} must be positive")
        if args.command == "train":
            if (Path(args.model_dir) / "model.json").exists():
                raise FileExistsError("Model already exists; choose a new --model-dir")
            from .classifier import configure_tensorflow
            configure_tensorflow("cpu" if args.architecture == "yolo11n" else args.device, args.threads, args.seed)
            bundle = dataset.load(args.data_dir, name=args.name)
            clf = CNNClassifier(bundle.class_names, args.architecture, args.image_size,
                                args.batch_size, args.learning_rate, args.seed, args.device,
                                args.threads, not args.no_augment, args.weights)
            val = list(dataset.iter_numpy(bundle["val"], args.eval_limit))
            clf.fit(dataset.iter_numpy(bundle["train"], args.limit), val, args.epochs,
                    args.patience, args.out_dir, verbose=0 if args.quiet else 2)
            clf.save(args.model_dir)
            if args.figures and clf.architecture == "resnet18":
                fig = visualize.plot_history(clf.history)
                visualize.save_figure(fig, Path(args.out_dir) / "learning_curves.png")
                plt.close(fig)
            if not args.skip_eval:
                scores = clf.predict_proba([x for x, _ in val])
                summary = save_evaluation([y for _, y in val], scores, clf.class_names, args.out_dir)
                print(f"Validation accuracy={summary['accuracy']:.4f} macro-F1={summary['macro_f1']:.4f}")
        else:
            clf = CNNClassifier.load(args.model_dir, device=args.device)
            if args.command == "info":
                print((Path(args.model_dir) / "model.json").read_text())
                print(f"Model size: {sum(p.stat().st_size for p in Path(args.model_dir).rglob('*') if p.is_file()) / 1e6:.3f} MB")
            elif args.command == "evaluate":
                bundle = dataset.load(args.data_dir, name=args.name)
                if bundle.class_names != clf.class_names:
                    raise ValueError("Dataset and model class order differ")
                samples = list(dataset.iter_numpy(bundle[args.split], args.limit))
                summary = save_evaluation([y for _, y in samples], clf.predict_proba([x for x, _ in samples]),
                                          clf.class_names, args.out_dir)
                print(f"{args.split}: accuracy={summary['accuracy']:.4f} macro-F1={summary['macro_f1']:.4f}")
            elif args.command == "predict":
                if args.true_label is not None and not 0 <= args.true_label < clf.n_classes:
                    raise ValueError("--true-label outside model class range")
                image = load_image(args.image)
                pred = clf.predict(image)
                payload = pred.to_dict(clf.class_names)
                print(json.dumps(payload, indent=2))
                out = Path(args.out_dir) / Path(args.image).stem
                out.mkdir(parents=True, exist_ok=True)
                (out / "prediction.json").write_text(json.dumps(payload, indent=2))
                if not args.no_viz:
                    figs = [("01_result.png", visualize.plot_prediction_summary(image, pred, clf.class_names, args.true_label))]
                    heat = (visualize.activation_map(clf, image) if clf.architecture == "resnet18"
                            else visualize.occlusion_map(clf, image))
                    figs.append(("02_explanation.png", visualize.plot_explanation(image, heat,
                                 "CAM" if clf.architecture == "resnet18" else "Oclusão: queda do escore")))
                    for name, fig in figs:
                        visualize.save_figure(fig, out / name)
                        if args.show:
                            plt.show()
                        plt.close(fig)
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"cnn: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
