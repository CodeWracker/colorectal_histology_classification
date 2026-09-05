# Inference benchmark — polygarbor

Measured at 2026-09-05T15:39:32+00:00 (UTC).

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

## Total time per image

Cold totals include Python startup and loading the model; warm totals are what you pay per image once the model is in memory.

| total | mean | median | stdev | best | worst | runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold total: new process, import, load model, classify | 1.86 s | 1.74 s | 293.3 ms | 1.68 s | 2.38 s | 5 |
| Cold total: `polygarbor predict --no-viz` command | 2.00 s | 1.73 s | 518.2 ms | 1.67 s | 2.60 s | 3 |
| Cold total: `polygarbor predict` with all figures | 5.62 s | 5.18 s | 781.4 ms | 5.16 s | 6.52 s | 3 |
| Warm total: read file and classify (model in memory) | 20.9 ms | 20.8 ms | 0.5 ms | 20.4 ms | 25.9 ms | 200 |
| Warm total: classify an array already in memory | 20.1 ms | 20.1 ms | 0.4 ms | 18.9 ms | 22.8 ms | 200 |
| Warm total, dense mode (similarity maps) | 965.2 ms | 942.1 ms | 58.1 ms | 885.2 ms | 1.20 s | 200 |

## Stage breakdown

| step | mean | median | stdev | best | worst | runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Load model from disk (rebuilds the subspaces) | 573.9 ms | 569.9 ms | 43.3 ms | 506.4 ms | 701.0 ms | 20 |
| Read and decode the image | 0.5 ms | 0.5 ms | 0.1 ms | 0.5 ms | 2.4 ms | 200 |
| Extract descriptor (patch) | 4.4 ms | 4.3 ms | 0.2 ms | 4.1 ms | 6.7 ms | 200 |
| Mahalanobis distances (patch) | 16.3 ms | 16.2 ms | 0.3 ms | 15.8 ms | 19.2 ms | 200 |
| Extract descriptor (dense 75x75) | 1.6 ms | 1.6 ms | 0.0 ms | 1.5 ms | 1.9 ms | 200 |
| Mahalanobis distances (dense) | 958.8 ms | 942.3 ms | 57.5 ms | 877.0 ms | 1.15 s | 200 |

## Reading the numbers

- **Warm: 20.9 ms per image** (~48 images/s) reading the file from disk, 20.1 ms if the array is already in memory. This is the number that matters for a service.
- **Cold: 1.86 s end to end** for a fresh process. Almost all of it is fixed overhead — 573.9 ms to rebuild the subspaces plus interpreter startup and imports — so classifying one image per process wastes 99% of the time on setup.
- `dense` mode costs 965.2 ms, ~48× the patch mode: it evaluates 5625 vectors instead of 1. That is the price of the similarity maps, not of the decision.
- The full `predict` command with every figure takes 5.62 s; rendering the five matplotlib figures dominates it, not the classification.

## How to reproduce

```bash
cd polygarbor
uv run python benchmarks/bench_inference.py \
  --model-dir artifacts/model \
  --image artifacts/dataset/samples/00_tumor_1.png
```

The script writes this markdown plus a JSON with the same numbers to
`benchmarks/results/`.
