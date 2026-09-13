"""Render saved CNN heatmaps with explicit signed scales, without inference."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cnn/src"))
from cnn import visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    visualize.use_headless()
    images = np.load(ROOT / "cache/test_images.npy", mmap_mode="r")
    rows = []
    for folder in (ROOT / "runs" / args.campaign).glob("full__*__seed*"):
        config = json.loads((folder / "config.json").read_text())
        if config["method"] not in ("resnet18", "yolo11n"):
            continue
        for path in (folder / "explanations").glob("*/heatmap.npy"):
            index = int(path.parent.name.split("_")[1])
            heat = np.load(path)
            title = "CAM — escala relativa" if config["method"] == "resnet18" else "Oclusão: queda de probabilidade"
            fig = visualize.plot_explanation(images[index], heat, title)
            visualize.save_figure(fig, path.parent / "explanation.png")
            plt.close(fig)
            rows.append(dict(run=folder.name, example=index, min=float(heat.min()), max=float(heat.max()),
                             zero_map=bool(np.all(heat == 0))))
    (ROOT / "explanation_ranges.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
