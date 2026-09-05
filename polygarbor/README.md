# polygarbor

Colorectal histology classification combining a **Gabor filter bank** (texture)
plus **Lab color statistics** as the descriptor, and a **polynomial Mahalanobis
subspace per class** as the classifier.

This is the organized, tested and runnable version of the
`garbor-downsampling.ipynb` prototype.

---

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and lives entirely inside this
folder.

```bash
cd polygarbor

# inference only (light, no TensorFlow)
uv sync

# + dataset flow (pulls TensorFlow CPU and TFDS)
uv sync --extra dataset
```

Every command then runs through `uv run`:

```bash
uv run polygarbor --help
```

---

## The two flows

### Flow 1 — load the dataset (once, permanently)

Downloads `colorectal_histology` (5000 images of 150×150 px, 8 classes) and
materializes it in the folder you choose. The cache is permanent: later runs
only read from disk.

```bash
uv run polygarbor dataset \
  --data-dir ./data \
  --preview 9 \
  --export-per-class 1
```

Reading is deterministic by default (the file order changes which vectors end up
in the training sample); pass `--shuffle` if you want a varied preview.

| option | effect |
| --- | --- |
| `--data-dir` | permanent TFDS folder (default `data`) |
| `--preview N` | saves a grid with N labeled samples |
| `--export-per-class N` | writes N PNGs per class to try out `predict` |
| `--export-split` | which split to export from (`train`/`val`/`test`) |
| `--shuffle` | shuffles the read order (default: deterministic) |

Then train the subspaces:

```bash
uv run polygarbor train --data-dir ./data --model-dir ./artifacts/model --figures
```

This extracts the training descriptors, fits one `PolyMahalanobis` per class,
saves the model and evaluates on the validation split (report and confusion
matrix as CSV and PNG). Useful knobs: `--patches-per-row`, `--levels`,
`--max-samples`, `--limit` (runs on a handful of images for a quick check) and
`--skip-eval`.

Standalone evaluation of a saved model:

```bash
uv run polygarbor evaluate --model-dir ./artifacts/model --split test
```

### Flow 2 — classify one image and generate the visualizations

```bash
uv run polygarbor predict \
  -i artifacts/dataset/samples/00_tumor_1.png \
  --model-dir ./artifacts/model \
  --out-dir ./artifacts/predict \
  --true-label 0        # optional: highlights hit/miss
```

Terminal output: predicted class, confidence and the full ranking (similarity,
votes and mean distance per class). Under `--out-dir/<image-name>/`:

| file | content |
| --- | --- |
| `prediction.json` | full result, ready to be consumed by another program |
| `01_result.png` | image + class ranking |
| `02_gabor_bank.png` | the kernels of the bank |
| `03_patch_decomposition.png` | patch → grayscale → Gabor energy maps |
| `04_descriptor_matrix.png` | heatmap of the matrix (patches × features) |
| `05_similarity_maps.png` | dense per-class similarity maps + winner per region |

`--no-viz` classifies without producing figures; `--show` opens the windows
instead of only writing the PNGs.

---

## Use as a library

The package is importable and **does not require TensorFlow** to classify — TFDS
is only loaded on demand by the dataset flow.

```python
from polygarbor import PolyGaborClassifier

clf = PolyGaborClassifier.load("artifacts/model")

pred = clf.predict("my_slide.png")            # path, ndarray or tensor
print(pred.class_name, f"{pred.confidence:.1%}")
print(pred.ranking(clf.class_names))          # [(class, similarity, votes, distance), ...]
print(pred.to_dict(clf.class_names))          # JSON-serializable dict
```

Training on your own data (any iterable of `(image, label)`):

```python
clf = PolyGaborClassifier(class_names=["a", "b"], patches_per_row=3)
clf.fit([(img1, 0), (img2, 1), ...])
clf.save("my_model")
```

Generating the figures programmatically:

```python
from polygarbor import visualize

visualize.use_headless()                      # only write PNGs, never open a window
pred = clf.predict(img, method="dense")       # the dense mode carries the grid
fig = visualize.plot_similarity_maps(img, pred, clf.class_names)
visualize.save_figure(fig, "maps.png")
```

Main API surface:

| symbol | role |
| --- | --- |
| `PolyGaborClassifier` | full pipeline: `fit`, `predict`, `predict_many`, `save`, `load` |
| `Prediction` | result: `label`, `class_name`, `confidence`, `votes`, `similarity`, `distances` |
| `GaborConfig` / `GaborBank` | filter bank parameters and construction |
| `patch_features` / `dense_features` | descriptor extraction outside the classifier |
| `evaluate` / `EvaluationResult` | accuracy, report and confusion matrix |

---

## How it works

1. **Global filtering.** The whole image is converted to grayscale and convolved
   with each Gabor kernel pair (real and imaginary). The energy is the magnitude
   of the pair. Filtering the whole image before cropping avoids border
   artifacts in the patches.
2. **22-dimensional descriptor.** Per region: mean and standard deviation of each
   of the 8 energy maps (16 values) + mean and standard deviation of the 3 Lab
   channels (6 values).
3. **One subspace per class.** Each class gets a `PolyMahalanobis` fitted on up
   to `--max-samples` vectors, with a polynomial expansion of `--levels` levels.
4. **Decision.** Each unit (patch or pixel) votes for its nearest class; the
   image takes the most voted class (`--aggregation voting`) or the one with the
   smallest mean distance (`--aggregation mean`).

### `patch` vs `dense`

These are two spatial scales of the **same** descriptor:

- `--method patch` (default) uses the patch grid — the same scale as training,
  which is why it is the right mode for deciding the class.
- `--method dense` computes the statistics per pixel with a sliding window on a
  75×75 grid. Its local scale differs from the one seen during training, so it
  works well to **visualize where** the image looks like each class, but tends to
  be less faithful as a final decision. `predict` always produces the dense maps,
  even when the decision comes from `patch` mode.

With `--patches-per-row 1` (the default, faithful to the prototype) there is a
single patch per image: the vote is trivial and confidence is always 100%. Use
`--patches-per-row 3` for the vote across patches to become meaningful.

---

## Saved model format

```
artifacts/model/
├── model.json            # configuration, classes and feature names
└── samples/class_NN.txt  # training vectors of each class
```

The subspaces are **rebuilt** from the samples inside `load()` — there is no
pickle, so the model stays readable and portable across library versions.

---

## Reference results

Default configuration (`--patches-per-row 1`, 8 filters, 3 levels, 350 samples
per class), trained on the 4000 training examples:

| split | accuracy |
| --- | --- |
| validation (500 images) | 0.828 |
| test (500 images) | 0.786 |

`adipose` (recall 1.00), `lympho` (0.94) and `tumor` (0.87) are the strongest
classes. The two weak ones are `mucosa` (recall 0.45, confused with `debris`)
and `complex` (0.51, confused with `stroma`) — both with high precision, meaning
the model is conservative about assigning them.

Since training is deterministic (no shuffling, fixed `--seed`), repeating the
command reproduces exactly the same model.

---

## Inference benchmark

Classifying one image takes **~20 ms** (~49 images/s) on the reference machine
(Intel Core Ultra 9 185H, CPU only). The `dense` mode costs ~960 ms because it
evaluates 5625 vectors instead of 1. Loading the model costs ~690 ms once per
process, so a batch service should load once and reuse the instance.

Full numbers and machine specs live in
[`benchmarks/results/`](benchmarks/results/). To measure on another machine:

```bash
uv run python benchmarks/bench_inference.py \
  --model-dir artifacts/model \
  --image artifacts/dataset/samples/00_tumor_1.png
```

---

## Tests

```bash
uv run pytest
```

The suite uses synthetic images and covers the descriptor, training, the
`save`/`load` round trip, prediction in both modes and the generation of every
figure — without needing the dataset or TensorFlow.

---

## Layout

```
src/polygarbor/
├── gabor.py       # GaborConfig and GaborBank
├── features.py    # patch and dense descriptors
├── classifier.py  # PolyGaborClassifier and Prediction
├── dataset.py     # permanent TFDS loading (lazy import)
├── evaluation.py  # per-image metrics
├── visualize.py   # every figure
├── console.py     # formatted terminal output
└── cli.py         # argparse: dataset / train / evaluate / predict / info

benchmarks/
├── bench_inference.py  # inference timing + machine specs
└── results/            # committed reports, one per machine
```
