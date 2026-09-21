# Setup: environments, hardware and data

## Reference machine

Every timing, latency and memory figure in the paper was measured on one laptop: an Intel Core Ultra 9 185H (16 cores, 22 logical processors), 32 GB of RAM and an NVIDIA RTX 4070 Laptop GPU with 8 GB, running Linux. The descriptor-based methods never touch the GPU. The CNNs were trained on the GPU in the registered campaigns and retrained on the CPU afterwards, with four threads, so that training cost could be compared on the same processor.

Accuracy results do not depend on this machine; cost results do. If you re-measure timings elsewhere, nothing else should be running, because the runners record wall-clock time, CPU and memory per stage.

## Disk budget

| Item | Size | Versioned |
| --- | ---: | --- |
| TFDS `colorectal_histology` cache (`polygarbor/data/`) | ~750 MB | no |
| Shared array cache (`comparison/cache/`) | ~340 MB | no |
| NCT replication cache (`comparison/cache/nct/`) | ~1.5 GB | no |
| Per-run artifacts (`comparison/runs/`) | tens of GB | no |
| Virtual environments (three of them) | ~10 GB | no |
| Everything in the repository | ~30 MB plus the two notebooks | yes |

Plan for roughly 60 GB of free space to rerun all campaigns from scratch. The YOLO LOSO runs delete their `yolo_dataset` image copy on completion, as registered in the protocol, which is what keeps that number from growing further.

## Python environments

The projects use [uv](https://docs.astral.sh/uv/) and target Python 3.11 (`polygarbor/.python-version`); `polygarbor` and `cnn` accept 3.10 to 3.12, while the quantization project requires exactly 3.11. There are three independent environments, created from committed lockfiles, and each is used by a fixed set of scripts.

```bash
# 1. descriptor + PMD, with the TFDS dataset flow            -> polygarbor/.venv
uv sync --project polygarbor --extra dataset

# 2. CNNs, GPU libraries and YOLO; also the environment that runs every
#    script under comparison/ except the two post-hoc PolyGabor ones  -> cnn/.venv
uv sync --project cnn --extra gpu --extra yolo

# 3. post-training quantization, kept apart on purpose       -> comparison/quantization/.venv
uv sync --project comparison/quantization
```

The `gpu` extra installs the CUDA 12 libraries through pip; `yolo` brings Ultralytics with PyTorch 2.7.1 and torchvision 0.22.1, which is also what the ImageNet ResNet port reads. TensorFlow 2.20 and PyTorch 2.7.1 are pinned together so both find the same CUDA 12 runtime. If TensorFlow does not see the GPU after the sync, run `cd cnn && uv run python benchmarks/setup_cuda.py`, which creates the local links described in the official TensorFlow installation guide; it changes neither the driver nor any system library.

The quantization environment is separate because `litert-torch` needs PyTorch 2.11, which would change the versions pinned for the training campaign. Models trained under 2.7.1 load into it unchanged, and the conversion is only accepted if the float32 LiteRT variant agrees with the saved predictions on at least 99% of the clean test set.

Scripts are invoked through the interpreter of the environment they belong to, from the repository root, for example `cnn/.venv/bin/python comparison/suite.py ...`. This matters: `comparison/suite.py` derives `LD_LIBRARY_PATH`, `PYTHONPATH`, `KERAS_HOME` and `YOLO_CONFIG_DIR` from the location of `cnn/.venv`, and passes them to every child process. The file header of each script states which environment it expects.

## Main dataset

```bash
polygarbor/.venv/bin/polygarbor dataset --data-dir polygarbor/data --preview 9 --export-per-class 1
```

This downloads TFDS `colorectal_histology` 2.0.0 once and materializes it permanently: 5,000 RGB patches of 150×150 pixels from ten H&E colorectal slides, 625 per class over eight classes, in the library's deterministic 4,000 / 500 / 500 split. Later runs only read from that directory. `--export-per-class` writes one PNG per class so that `predict` can be tried immediately.

Do not delete `polygarbor/data/downloads/extracted/`: `comparison/prepare_groups.py` recovers the slide source of every patch by matching SHA-256 pixel hashes against those original `.tif` files, whose names carry the `CRC-Prim-HE-01` to `-10` codes. Without them the LOSO experiment cannot be rebuilt.

The patches are then materialized as shared NumPy arrays, identical for every method:

```bash
cnn/.venv/bin/python comparison/prepare.py          # cache/{train,val,test}_{images,labels}.npy + dataset_manifest.json
cnn/.venv/bin/python comparison/prepare_groups.py   # source_manifest.json
```

`dataset_manifest.json` records the class, per-class counts and the SHA-256 of every image in split order, and asserts that no exact duplicate crosses split boundaries. Both manifests are versioned here, so a fresh preparation can be compared against them byte for byte.

## Replication dataset

```bash
cnn/.venv/bin/python comparison/nct_prepare.py
```

This prepares the NCT-CRC-HE-100K → CRC-VAL-HE-7K replication from Zenodo record 1214456, in the color-normalized version distributed there. The 11.7 GB training archive is never downloaded whole: the script reads its central directory with HTTP range requests and fetches only the 1,347 members needed for the validation set and the selected training images, checking each against its CRC-32. The 0.8 GB test archive is downloaded in full, because all 7,180 test patches are used. Every patch is center-cropped to 150×150 pixels (offset 37) so the descriptor sees roughly the same physical field as on the main dataset.

`nct_prepare.py` refuses to run once `cache/nct/manifest.json` exists, because the selection is fixed and must never be redrawn; delete the manifest only if you intend to start the replication over. The script is polite to Zenodo by design, at about one request per second; the first attempt during the study, with 16 unthrottled threads, was cut off by HTTP 429. Expect the preparation to take a while, and note that it is resumable: cached members are reused and the selection is deterministic, so an interrupted run can simply be restarted.

`cache/nct/manifest.json` records the file names, SHA-256 hashes and the fixed splits: 50 validation patches per class (450 total), nested training budgets of 1, 2, 5, 10 and 20 images per class for seeds 42 to 46, and the whole 7,180-patch test set.

## YOLO weights

`yolo11n-cls.pt` is downloaded automatically by Ultralytics on first use and lands at the repository root, where `.gitignore` excludes it. The `yolo11n_random` variant needs no download: it builds the architecture from `yolo11n-cls.yaml` with `pretrained=False`.
