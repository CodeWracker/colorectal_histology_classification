# Inference benchmark — polygarbor

Measured at 2026-09-05T15:21:49+00:00 (UTC).

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

## What was measured

| item | value |
| --- | --- |
| Image | `artifacts/dataset/samples/00_tumor_1.png` (150x150 px) |
| Model | `artifacts/model` — 8 classes, 3 polynomial levels |
| Descriptor | 22D |
| Vectors per image | 1 (patch) / 5625 (dense) |
| Repetitions | 200 measured after 20 warm-up runs |
| Predicted class | tumor |

## Results

| step | mean (ms) | median (ms) | stdev (ms) | best (ms) | worst (ms) | per second |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Load model from disk (once per process) | 691.98 | 666.37 | 66.17 | 588.19 | 823.04 | 1.4 |
| Read and decode the image | 0.50 | 0.50 | 0.01 | 0.48 | 0.54 | 2017.7 |
| Extract descriptor (patch) | 4.49 | 4.43 | 0.17 | 4.29 | 5.33 | 222.8 |
| Mahalanobis distances (patch) | 16.42 | 16.33 | 0.42 | 15.85 | 19.19 | 60.9 |
| Patch inference, in-memory array | 20.39 | 20.29 | 0.48 | 19.74 | 25.09 | 49.1 |
| Patch inference, from file | 20.65 | 20.61 | 0.28 | 20.14 | 21.68 | 48.4 |
| Extract descriptor (dense 75x75) | 1.53 | 1.51 | 0.05 | 1.47 | 1.72 | 653.5 |
| Mahalanobis distances (dense) | 955.88 | 937.98 | 58.04 | 876.16 | 1169.65 | 1.0 |
| Dense inference, in-memory array | 961.17 | 943.14 | 57.83 | 882.93 | 1221.48 | 1.0 |

## Reading the numbers

- **Classifying one image costs 20.4 ms** (~49 images/s) in `patch` mode, which is the default. Including reading the file from disk, 20.7 ms.
- `dense` mode costs 961.2 ms, ~47× more: it evaluates 5625 vectors instead of 1. That is the price of the similarity maps.
- Loading the model takes 692 ms, but it is a one-off cost per process: the subspaces are rebuilt from the samples inside `load()`. A service classifying in batch should load once and reuse the instance.

## How to reproduce

```bash
cd polygarbor
uv run python benchmarks/bench_inference.py \
  --model-dir artifacts/model \
  --image artifacts/dataset/samples/00_tumor_1.png
```

The script writes this markdown plus a JSON with the same numbers to
`benchmarks/results/`.
