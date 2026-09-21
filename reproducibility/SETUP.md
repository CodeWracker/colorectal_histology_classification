# Setup

## Reference machine

All timings, latencies and memory figures come from one laptop: Intel Core Ultra 9 185H (16 cores, 22 logical processors), 32 GB RAM, NVIDIA RTX 4070 Laptop with 8 GB, on Linux. The descriptor methods never touch the GPU. The CNNs trained on the GPU in the registered campaigns and were retrained on the CPU afterwards with four threads, so training cost could be compared on the same processor.

Accuracy does not depend on this machine. Cost does. If you re-measure timings, run nothing else at the same time, since the runners record wall-clock time, CPU and memory per stage.

## Disk

| Item | Size | Versioned |
| --- | ---: | --- |
| TFDS cache, `polygarbor/data/` | ~750 MB | no |
| Array cache, `comparison/cache/` | ~340 MB | no |
| Replication cache, `comparison/cache/nct/` | ~1.5 GB | no |
| Per-run artifacts, `comparison/runs/` | tens of GB | no |
| Three virtual environments | ~10 GB | no |
| The repository itself | ~30 MB plus the two notebooks | yes |

Plan for about 60 GB free to rerun everything. The YOLO LOSO runs delete their `yolo_dataset` image copy on completion, as registered, which keeps that number from growing further.

## Environments

The projects use [uv](https://docs.astral.sh/uv/) and target Python 3.11 (`polygarbor/.python-version`). `polygarbor` and `cnn` accept 3.10 to 3.12; the quantization project requires exactly 3.11. There are three environments, built from committed lockfiles, and each one owns a fixed set of scripts.

```bash
# descriptor and PMD, with the TFDS dataset flow           -> polygarbor/.venv
uv sync --project polygarbor --extra dataset

# CNNs, CUDA and YOLO; also runs every comparison/ script
# except the two post-hoc PolyGabor ones                   -> cnn/.venv
uv sync --project cnn --extra gpu --extra yolo

# post-training quantization, deliberately separate        -> comparison/quantization/.venv
uv sync --project comparison/quantization
```

The `gpu` extra installs the CUDA 12 libraries through pip. `yolo` brings Ultralytics with PyTorch 2.7.1 and torchvision 0.22.1, which is also what the ImageNet ResNet port reads. TensorFlow 2.20 and PyTorch 2.7.1 are pinned together so both find the same CUDA 12 runtime. If TensorFlow does not see the GPU after the sync, run `cd cnn && uv run python benchmarks/setup_cuda.py`, which creates the local links from the official TensorFlow installation guide. It changes no driver and no system library.

Quantization is separate because `litert-torch` needs PyTorch 2.11, which would move versions pinned for the training campaign. Models trained under 2.7.1 load into it unchanged, and a conversion is only accepted if the float32 LiteRT variant agrees with the saved predictions on at least 99% of the clean test set.

Scripts run through the interpreter of the environment they belong to, from the repository root, for example `cnn/.venv/bin/python comparison/suite.py`. This matters: `suite.py` derives `LD_LIBRARY_PATH`, `PYTHONPATH`, `KERAS_HOME` and `YOLO_CONFIG_DIR` from the location of `cnn/.venv` and passes them to every child process. Each script's header states which environment it expects.

## Main dataset

```bash
polygarbor/.venv/bin/polygarbor dataset --data-dir polygarbor/data --preview 9 --export-per-class 1
```

Downloads TFDS `colorectal_histology` 2.0.0 once and materializes it permanently: 5,000 RGB patches of 150x150 pixels from ten H&E colorectal slides, 625 per class over eight classes, in the library's deterministic 4,000 / 500 / 500 split. Later runs only read from that directory. `--export-per-class` writes one PNG per class so `predict` can be tried right away.

Keep `polygarbor/data/downloads/extracted/`. `prepare_groups.py` recovers the slide source of every patch by matching SHA-256 pixel hashes against those original `.tif` files, whose names carry the `CRC-Prim-HE-01` to `-10` codes. Without them the LOSO experiment cannot be rebuilt.

Then materialize the shared arrays, identical for every method:

```bash
cnn/.venv/bin/python comparison/prepare.py          # cache arrays + dataset_manifest.json
cnn/.venv/bin/python comparison/prepare_groups.py   # source_manifest.json
```

`dataset_manifest.json` records the class, per-class counts and SHA-256 of every image in split order, and asserts that no exact duplicate crosses a split boundary. Both manifests are versioned, so a fresh preparation can be compared against them byte for byte.

## Replication dataset

```bash
cnn/.venv/bin/python comparison/nct_prepare.py
```

Prepares the NCT-CRC-HE-100K to CRC-VAL-HE-7K replication from Zenodo record 1214456, color-normalized version. The 11.7 GB training archive is never downloaded whole: the script reads its central directory with HTTP range requests and fetches only the 1,347 members needed for validation and for the selected training images, checking each against its CRC-32. The 0.8 GB test archive is downloaded in full, since all 7,180 test patches are used. Every patch is center-cropped to 150x150 pixels (offset 37) so the descriptor sees roughly the same physical field as on the main dataset.

The script refuses to run once `cache/nct/manifest.json` exists, because the selection is fixed and must not be redrawn. Delete that manifest only to start the replication over. It runs at about one request per second by design, after the first attempt with 16 unthrottled threads was cut off by HTTP 429, and it caches every fetched member, so an interrupted preparation can be restarted.

`cache/nct/manifest.json` records file names, SHA-256 hashes and the fixed splits: 50 validation patches per class, nested training budgets of 1, 2, 5, 10 and 20 images per class for seeds 42 to 46, and the whole 7,180-patch test set.

## YOLO weights

Ultralytics downloads `yolo11n-cls.pt` on first use, to the repository root, where `.gitignore` excludes it. The `yolo11n_random` variant needs no download: it builds the architecture from `yolo11n-cls.yaml` with `pretrained=False`.
