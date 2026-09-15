# Training benchmark — polygarbor

Measured at 2026-09-05T15:41:46+00:00 (UTC).

## Machine

| item | value |
| --- | --- |
| CPU | Intel(R) Core(TM) Ultra 9 185H |
| Cores / threads | 16 cores × 2 thread(s) = 22 logical CPUs |
| Max frequency | 5100.0000 MHz |
| Memory | 30.9 GB |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB |
| System | Linux 6.17.0-14-generic (x86_64) |
| Python | 3.11.14 |
| OpenCV threads | 22 |

> The GPU is listed only to describe the machine: the pipeline is
> CPU-only (OpenCV + NumPy) and does not use GPU acceleration.

### Versions

| package | version |
| --- | --- |
| numpy | 2.4.6 |
| opencv-python-headless | 5.0.0.93 |
| polymahalanobis | 0.1.1 |
| scikit-learn | 1.9.0 |
| tensorflow-cpu | 2.21.0 |
| tensorflow-datasets | 4.9.10 |

## What was measured

| item | value |
| --- | --- |
| Dataset | `data` — 4000 training images, 8 classes |
| Patch grid | 1x1 (1 vector(s) per image) |
| Vectors extracted | 4000 of 22D |
| Vectors used per subspace | up to 350 |
| Polynomial levels | 3 |
| Repetitions | 3 full training runs |

## Total training time

| total | mean | median | stdev | best | worst | runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Total training time | 18.84 s | 18.26 s | 1.11 s | 18.13 s | 20.12 s | 3 |

## Stage breakdown

| step | mean | median | stdev | best | worst | runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Load the dataset (TFDS, already materialized) | 664.8 ms | 112.1 ms | 985.2 ms | 79.9 ms | 1.80 s | 3 |
| Extract descriptors from the training split | 17.58 s | 17.55 s | 72.6 ms | 17.52 s | 17.66 s | 3 |
| Fit the polynomial subspaces (one per class) | 569.3 ms | 566.4 ms | 69.0 ms | 501.8 ms | 639.7 ms | 3 |
| Save the model to disk | 20.5 ms | 19.3 ms | 2.1 ms | 19.2 ms | 22.8 ms | 3 |

## Reading the numbers

- **Training the whole model takes 18.84 s** on 4000 images.
- Feature extraction is 93% of that (17.58 s): 4.4 ms per image, about 228 images/s. It is the only stage that grows with the dataset, and it is embarrassingly parallel — nothing here is parallelized yet.
- Fitting the 8 subspaces takes only 569.3 ms, because each one sees at most 350 vectors of 22 dimensions. Raising `--max-samples` is cheap; raising the number of images is not.
- Opening the already-materialized dataset costs 1.80 s on the first run and 112.1 ms afterwards: the expensive part is importing TensorFlow once per process, not reading the data.
- These numbers cover training only. The `train` command also evaluates the validation split afterwards, which the inference benchmark covers separately.

## How to reproduce

```bash
cd polygarbor
uv sync --extra dataset
uv run polygarbor dataset --data-dir data   # once
uv run python benchmarks/bench_training.py --data-dir data
```

The script writes this markdown plus a JSON with the same numbers to
`benchmarks/results/`.
