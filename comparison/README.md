# Reproducible experiments

The implementation is in `../cnn`, and the original method remains in `../polygarbor`. The [protocol](PROTOCOL.md) records the decisions made before the experiments. Start with the [discussion and visual assessment](results/2026-09-12/DISCUSSION.md). The consolidated results are in [results/2026-09-12/REPORT.md](results/2026-09-12/REPORT.md); each row of the index points to the corresponding run.

Run from the repository root:

```bash
uv sync --project cnn --extra gpu --extra yolo
cnn/.venv/bin/python comparison/prepare.py
cnn/.venv/bin/python comparison/prepare_groups.py
cnn/.venv/bin/python comparison/suite.py --campaign my-campaign --methods polygarbor polygarbor_aug polygarbor_p3 polygarbor_p3_aug polygarbor_p5 resnet18 resnet18_imagenet yolo11n yolo11n_random
cnn/.venv/bin/python comparison/loso.py
cnn/.venv/bin/python comparison/suite.py --campaign my-campaign --methods polygarbor resnet18 resnet18_imagenet yolo11n yolo11n_random --scenarios loso_01 loso_02 loso_03 loso_04 loso_05 loso_06 loso_07 loso_08 loso_09 loso_10
cnn/.venv/bin/python comparison/finish.py --campaign my-campaign
cnn/.venv/bin/python comparison/loso_report.py --campaign my-campaign
uv sync --project comparison/quantization
comparison/quantization/.venv/bin/python comparison/quantization/quantize.py --campaign my-campaign
comparison/quantization/.venv/bin/python comparison/quantization/bench.py --campaign my-campaign
comparison/quantization/.venv/bin/python comparison/quantization/report.py --campaign my-campaign
cnn/.venv/bin/python comparison/embedded_feasibility.py --model comparison/runs/my-campaign/full__polygarbor__seed42/model --output comparison/results/my-campaign/embedded
```

`finish.py` also runs `localization.py`, which reuses the full models of both seeds to measure tumor identification in synthetic 4×4 mosaics. Each image is explained in isolation before assembly, which prevents mixing between patches. The primary evaluation uses per-patch AUROC/AP and recovery of the two tumors in the top-2; region Dice uses a threshold chosen on validation and is secondary because there are no internal masks. Version 3 compares PolyGabor, random ResNet and ImageNet ResNet and separates, in each figure, the classification scores from the explanation maps. To run only this stage after the full models exist, use `cnn/.venv/bin/python comparison/localization.py --campaign my-campaign`; `--refresh` regenerates metrics and figures from the saved maps and writes its telemetry to a separate subdirectory.

The runner sets up local paths for the CUDA libraries and starts one process per run. Runs are sequential to avoid competition between models. The official YOLO weights are downloaded on first use. `prepare.py` reuses the TFDS data already available in `polygarbor/data` and saves arrays to `comparison/cache`; it does not modify the original cache. `dataset_manifest.json` contains the class, count and SHA-256 of each image.

`suite.py` accepts the methods `polygarbor`, `polygarbor_aug`, `polygarbor_p3`, `polygarbor_p3_aug`, `polygarbor_p5`, `resnet18`, `resnet18_imagenet`, `yolo11n` and `yolo11n_random` through `--methods`, `--scenarios full few1 few2 few5 few10 few20 few50 few100 imbalance label_noise`, `--seeds 42 43` and `--epochs 100`. `--pilot --epochs 1 --scenarios few10 --seeds 42` checks the installation; use another campaign name so that these results do not enter the main comparison. Existing runs are preserved, including failures; to repeat a failure, use a new campaign. An external interruption of the runner can leave a `running` status: check the log and do not read it as success. When the runner sees a non-zero exit from the child process, it records the failure. `finish.py` propagates test and benchmark failures and only reuses measurements recorded as successful.

Full training and the seven smaller sizes use two seeds. Imbalance and label noise use the first seed. Each training run is evaluated on training, validation and test. Models trained on the full set receive eight additional test evaluations with fixed degradations. Configuration, selected indices, altered labels, code hashes, package versions, history, timings, resources, individual predictions, matrices and explanations are stored in the run directory. Large models and data remain on disk and are ignored by Git.

`finish.py` measures inference on CPU with one thread and restricted affinity, on CPU with four threads and on GPU, in fresh processes, using the full seed 42 models. It then runs the tests and generates the report. `report.py` can be run again at any time, since it only reads saved artifacts. Do not run additional tests or benchmarks in parallel with training if you are going to compare timings.

```text
comparison/
  PROTOCOL.md
  dataset_manifest.json
  cache/                            # shared arrays, ignored by Git
  runs/2026-09-12/
    events.jsonl                    # commands, exit codes, process duration
    full__resnet18__seed42/
      config.json / selection.json / status.json / run.log
      timings.json / resources.csv / resources.json
      model/ / model_size.json
      history.csv / history.json / epochs.jsonl
      eval/test_clean/metrics.json / predictions.npz / confusion_matrix.png
      explanations/
    edge/resnet18__cpu1/
  results/2026-09-12/
    REPORT.md / DISCUSSION.md
    run_index.csv / metrics.csv / learning_curve.csv / paired_tests.csv / initialization_comparison.csv
    learning_curve.png / learning_curve_gmean.png / robustness.png / tradeoffs.png / edge.png
    stage_resources.csv / edge.csv / aggregation_comparison.csv
    error_gallery.png / source_shift_gallery.png / corruption_gallery.png
    localization/README.md / metrics.csv / localization_metrics.png / localization_examples.png / localization_pretraining_examples.png
    embedded/README.md / feasibility.json / feasibility.csv
    quantization/README.md / metrics.csv / benchmark.csv / summary.csv / speed.csv / calibration.json / quantization_pareto.png
    loso/README.md / folds.csv / fold_metrics.csv / pooled_metrics.csv / pooled_recall_per_class.csv / paired_polygarbor_vs_cnn.csv / loso_macro_f1.png / loso_recall_per_class.png
```

Interpretation: the few-shot scenarios restrict training but keep 500 labeled examples for validation. The full PolyGabor point keeps the default cap of 350 vectors/class and records how many vectors enter the fit. ResNet-18 and YOLO11n have paired controls with random and ImageNet initialization. The CAM, occlusion and similarity maps are computational explanations, not segmentation. The edge simulation measures x86 with less parallelism; it does not demonstrate performance on ARM/Jetson. The source codes recovered from the file names allow grouped tests, but they do not clinically identify patients. The source analysis uses ten leave-one-source-out folds: `loso.py` reads `source_manifest.json` and checks the fixed manifest `loso_manifest.json`, and `loso_report.py` generates `results/<campaign>/loso/`. The `source_a` and `source_b` scenarios (`--scenarios source_a source_b --seeds 42`) remain reproducible only as a historical record; the [protocol](PROTOCOL.md) explains why they were replaced.

The `p3` and `p5` variants use 9 and 25 regions per image; the filter bank stays the same. `_aug` adds three views per original image, in training only. The cap of 350 vectors/class remains, and the number of originals actually retained is stored in `effective_training.json`. See [G-mean versus macro-F1](GMEAN_VS_MACRO_F1.md) to interpret the two curves without confusing zero recall with a training failure.

The totals and final state of the campaign are in [integrity_audit.json](integrity_audit.json) and in the generated report. The two runs interrupted when the folder was moved were archived and repeated successfully.

Post-training quantization converts the full ResNet-18 and YOLO11n models to LiteRT and produces float16 weights, dynamic int8 and static int8, without retraining; PolyGabor stays at full precision as a reference. It uses the separate environment `comparison/quantization`, because litert-torch requires torch 2.11 and would change the versions pinned in `cnn/`. `quantize.py` evaluates each variant on the clean test and the eight perturbations, `bench.py` measures latency and peak RAM in fresh processes in the cpu1 and cpu4 modes, and `report.py` generates `results/<campaign>/quantization/`. Do not run the benchmark together with training or with `quantize.py`.

The embedded preflight compares the trained PolyGabor model with the flash and SRAM limits of the ESP32-WROOM-32, Arduino Uno R3 and PIC16F877A. It accounts for the saved artifact, the rebuilt polynomial state, float32/int16/int8 estimates, image buffers and Gabor operations. It is a reproducible static analysis; instruction-level simulation and energy measurement depend on a C/C++ port and on hardware.
