# Work log

## Preparation

Inspected `cnn_test.ipynb`, the PolyGabor CLI and API, their tests and the existing artifacts. The notebook contains a ResNet-18 trained from scratch and a YOLO11n-cls with external weights. The local TFDS cache already had the 5000 images. Identified hardware: Intel Core Ultra 9 185H, 22 logical processors, approximately 32 GB RAM and an 8 GB NVIDIA RTX 4070 Laptop GPU shared with the desktop.

The sandbox could not start commands (`bwrap: loopback: Failed RTM_NEWADDR`). The authorized reads, installations and runs went through the alternative execution path with automatic review. This also affected the image viewer; the figures were inspected by reading and encoding a preview, keeping the original files intact.

The protocol was written before the main experiments and extended at the authors' request with a 1/2/5/10/20/50/100 examples-per-class curve and an edge simulation. From then on, Markdown was written with each paragraph on a single line. The splits were materialized as shared arrays in 5.33 s; a SHA-256 audit found no exact duplicates. The manifest records the 5000 images in order.

## Environment and pilots

Created an isolated environment in `cnn/.venv`. The initial resolution brought in mutually incompatible CUDA libraries; TensorFlow 2.20.0, PyTorch 2.7.1 and torchvision 0.22.1 were pinned, with a uv lockfile. The CUDA 12 libraries were reinstalled after removing CUDA 13 packages with overlapping files. Local links and the campaign's `LD_LIBRARY_PATH` solved library discovery, without changing the driver.

`pilot-v1`: PolyGabor few10/seed42 finished in 11.6 s. ResNet failed because TensorFlow did not yet detect the GPU. The failure is preserved with traceback and status; it did not enter the main comparison.

`pilot-v2`: ResNet few10/seed42/one epoch finished in 34.8 s and YOLO in 23.8 s, including the whole pilot pipeline. Both used the GPU. These numbers serve to verify that things work, not to compare convergence. The first ResNet epoch includes the initial compilation.

The initial small tests passed (4 tests). The first attempt to also collect the PolyGabor tests hit a collision between `test_pipeline.py` files; the final run uses `--import-mode=importlib`. That attempt failed at collection in 1.15 s, at the start of the campaign, before the main fitting. No tests or other training were kept running in parallel with the measurements.

## Independent review

The review pointed out that Ultralytics ignores `batch` for a whole NumPy list; inference now splits the list into batches explicitly. Also fixed: seed 43 missing from full training, and ResNet parameters that YOLO accepted without effect, which are now rejected when they do not apply. The default PolyGabor cap of 350 vectors/class was kept and the effective budget is now recorded explicitly.

A second review identified that Ultralytics internally resets the thread count and that the edge benchmark accessed images late through memmap. The thread limit is reapplied after the backend is initialized; the edge images are copied to RAM before measurement. The training and edge experiments remain sequential.

## Partial results

First full PolyGabor, seed42: test accuracy 0.786 and macro-F1 0.7802, reproducing the existing reference. Descriptor extraction: 20.72 s; fitting: 0.47 s; full package of evaluations and explanations: 249.5 s. The total cost includes evaluation on training, validation, nine test conditions and figures. Matrix and maps were already inspected: `complex` and `mucosa` have the lowest recalls, while the Gabor responses highlight local frequencies and orientations.

First full ResNet, seed42: early stopping after 26 epochs, restoring the best validation state; test accuracy 0.906 and macro-F1 0.9052. Validation oscillated considerably early in training. Performance dropped notably under color shift, showing that the clean-test advantage does not automatically carry over to every degradation. This observation did not lead to any hyperparameter tuning.

Later progress is recorded in `campaign.log` and `runs/2026-09-12/events.jsonl`; `epochs.jsonl`, `history.csv` and `timings.json` keep the partial stages of each run. The final tables are rebuilt directly from the metric files.

## Recovery after a directory move and metric extension

The repository was temporarily moved to another directory and then back. This interrupted two YOLO runs: label_noise and source_b. Their logs show FileNotFoundError when saving artifacts to the old absolute paths. The attempts were marked as interrupted and preserved in `runs/interrupted`, and only these two will be repeated. The audit checked 56 complete runs, including label correspondence and test order, metric consistency, presence of resources and model size; all passed.

The two few1 failures of the original PolyGabor are unrelated to that incident: the library raises IndexError when fitting a single row per class. They remain as observed failures of the implementation, without being replaced by zero F1/G-mean. The variants with patches and augmentation will test whether multiple descriptors derived from one image change this behavior.

G-mean was added and recomputed from the saved predictions, without retraining. The document `GMEAN_VS_MACRO_F1.md` distinguishes the geometric mean of recalls from the arithmetic mean of F1. The first plot revealed that ResNet can keep a zero G-mean with few examples even when macro-F1 is already rising. All 19 tests passed after fixing the import in importlib mode.

Four PolyGabor ablations were queued, with grids of 9/25 regions and deterministic augmentation. An independent review confirmed label alignment with the patches, reproducibility of the selection, preservation of the vote-based decision in the metrics and handling of zero/missing recall. The report now also shows how many distinct original images actually enter the fit after the vector cap.

## Campaign completion and artifact review

The two interrupted YOLO runs were repeated and finished successfully. The extension also completed the 80 runs of the four PolyGabor variants. Final result: 138 complete and audited runs, two known few1 failures of the original PolyGabor, 16 edge benchmarks and 16 cold subprocesses with return code zero. No training remains pending. The sum of process times over the 140 final events is approximately 3.50 hours; it does not include all preparation, pilots, interruptions or edge benchmarks.

The old ResNet model contained Adam moments because clone_model copied the compilation state. The models were exported as uncompiled Functional models, keeping the original checkpoints outside the inference directory. The export of the full seed42 training reduced the artifact to 45.009 MB and produced a maximum probability difference of zero; each conversion keeps inference_export.json. The edge measurements use the corrected inference models.

Matrices, curves, Gabor/CAM/occlusion maps and the error, perturbation and source galleries were inspected. Overlapping captions in the galleries were fixed; occlusion maps now show sign and magnitude, avoiding reading a nearly null score drop as strong evidence. The final discussion distinguishes observation, hypothesis and experimental limitation.

The final review found that finish.py could ignore a non-zero exit from tests/benchmarks. The runner now propagates failures, can complete a missing cold measurement without repeating a valid warm one, and the report excludes timings from failed subprocesses. The regression was verified: 20 tests passed. Eight real CLI commands also passed, including info/predict/evaluate for both CNNs, dataset help and PolyGabor info. The first smoke attempt mistakenly used the base interpreter when resolving the venv symlink; it was corrected to the absolute venv path, without changing the models. The final logs are in results/2026-09-12/cli_smoke.

Interpreted results are in results/2026-09-12/DISCUSSION.md; reproducible tables in REPORT.md; G-mean is explained separately in GMEAN_VS_MACRO_F1.md. stage_resources.csv consolidates the per-stage measurements. The Markdown files keep each paragraph on one line, as requested.

## Quantitative localization

A protocol was fixed before running, with 20 validation and 30 test mosaics, two tumor patches per 4×4 mosaic and no repetition within each split. The full models of seeds 42/43 were reused. Dice thresholds were chosen on validation only; AUROC, AP, Dice and IoU were computed on pixels, with bootstrap intervals over mosaics.

After inspecting the first figure, version 1 was replaced because maps computed on the whole mosaic crossed artificial borders. Version 2 explains each patch in isolation and reassembles the maps without mixing, marks the ground truth, the top-2 and the threshold in the figure, and uses the patch as the primary unit. The original version remains described in the protocol to record the revision.

PolyGabor obtained per-patch explanation AUROC 0.8274/0.8233 and top-2 tumor recall 46.7%/45.0%; ResNet obtained 0.9703/0.9860 and 78.3%/85.0%. Per-patch classifier AUROC was 0.9638/0.9641 for PolyGabor and 0.9981/0.9984 for ResNet, showing that turning the decision into a spatial explanation loses more information in PolyGabor. The maps, layouts, classification scores, thresholds, resources and metrics were saved in runs/2026-09-12/localization and results/2026-09-12/localization.

Visual inspection of version 2 confirmed that the contours do not computationally cross the borders. In the three fixed examples, ResNet recovered both tumors and PolyGabor recovered zero, one and zero. The test measures patch identification and weak concentration of evidence, not segmentation of continuous tissue. The local dataset does not contain `colorectal_histology_large`, so the qualitative extension on the ten large images remains explicitly separate and was not run.

## 2026-09-13 — initialization controls and localization v3

The variants `resnet18_imagenet` and `yolo11n_random` were added. The ResNet ports 20 convolutions and 20 BatchNorm sets from the Torchvision ImageNet V1 checkpoint to the Keras architecture, keeps a random eight-class head and fine-tunes all layers; the random YOLO starts from `yolo11n-cls.yaml` with `pretrained=False`. A few10/seed42 pilot confirmed the origin of the weights and serialization before the definitive campaign.

The 40 additional definitive runs finished: two architectures, full training and seven few-shot sizes on seeds 42/43, imbalance and label noise on seed 42, and source_a/source_b on seed 42. The campaign went from 138 to 178 valid runs. The sum of the 180 recorded training events, including the two PolyGabor few1 failures, is 16,002.7 s or approximately 4.45 hours; it does not include pilots, localization, edge or preparation.

On the full test, mean macro-F1 was 0.9561 for ImageNet ResNet versus 0.9078 random, and 0.9510 for ImageNet YOLO versus 0.8618 random. The transfer gain was larger with few examples. The clearest exception was ResNet on source_a, which dropped from 0.5407 to 0.4841 and had zero G-mean. The table `initialization_comparison.csv` holds all paired deltas per scenario.

Localization version 3 reused without recomputation the PolyGabor and random ResNet maps from version 2, preserved the previous metadata with the `_v2` suffix and computed ImageNet ResNet CAM on the same layouts. Its per-patch explanation AUROC was 0.9980/0.9964 and top-2 recall 98.3%/96.7%, above the 0.9703/0.9860 and 78.3%/85.0% of random ResNet. The figures now separate known ground truth, per-patch classification score and explanation map. The ResNet comparison measures the full transfer configuration, including ImageNet weights and normalization, and does not causally isolate the weights alone.

Six new edge benchmarks were run, for a total of 22 warm modes and 22 cold subprocesses. Initialization barely changed size or latency within each architecture. One prediction out of one hundred from ImageNet ResNet on GPU diverged from the saved prediction; it was an original near tie between complex 0.4186 and tumor 0.4169. CPU1 and CPU4 agreed fully. The final summary is now generated by `validate.py` and records this 99% agreement without requiring bitwise determinism across devices.

The suite before the review fixes passed with 21 tests and 108 dependency warnings; the final one, with the transfer test, passed with 22 tests and 210 warnings. The audit checked 178 runs and found 178 valid. The curve, robustness and edge plots were split into panels; trade-offs became a matrix aligned by method, and the new visualizations were inspected at reduced size before consolidation.

The final review found that the first localization `--refresh` had overwritten `timings.json` and `resources.json` with only the metric stages. Version 2 remained preserved in the `_v2` files, and the generation seconds remained in raw_metrics. The refresh now writes to a subdirectory; the ImageNet ResNet extension was measured again without retraining, and `telemetry_manifest.json` points to the v2 base and the v3 complement. The same review added a direct test of the Torchvision mapping and the ImageNet roundtrip, made the pytest summary reject lines containing failures and made missing CLI failures or edge modes fail validation.

## 2026-09-13 — microcontroller preflight

A reproducible static analysis was added for ESP32-WROOM-32, Arduino Uno R3 and PIC16F877A. It distinguished the 0.651 MB saved directory from the loaded state: the library rebuilds the bases and keeps the samples, totaling 3.857 MB in arrays. An inference-only export can drop the samples, but it was still estimated at 3.587 MB with float32 and 0.910 MB with int8 coefficients and uint16 indices.

The current OpenCV implementation requires at least 1.176 MB for explicit maps of a 150×150 image. A streaming port was estimated at 27.2 kB of workspace, including float32 polynomial temporaries, and the direct Gabor bank requires approximately 158.76 million multiply-accumulate operations per image. The ESP32 has a plausible route after porting, quantization and reading coefficients from flash; Uno and PIC cannot hold the current int8 state. On-device training was classified as infeasible on all three targets.

The script, JSON, CSV and report are in comparison/embedded_feasibility.py and results/2026-09-12/embedded. The analysis is a memory and operation preflight, not instruction, latency or energy emulation.

## 2026-09-13 — tests and leave-one-source-out generalization

Both packages had `tests/test_pipeline.py`. In importlib mode with the root as rootdir, pytest registered empty `cnn` and `polygarbor` packages that shadowed the packages in `src/`, and the suite only collected with absolute paths. The modules were given unique names and a root `pytest.ini` declares `testpaths` and `pythonpath`; plain `pytest` collects the 22 tests, and `finish.py` no longer uses the override. Environment records and the CLI smoke no longer contain local machine paths.

The review of `source_a`/`source_b` showed that they do not support comparison between methods: two manual partitions with one seed, validation without adipose and empty used by the CNNs' early stopping, and in `source_b` 42% of the test set was empty, a class with 35 examples in training. PolyGabor classified 99.8% of those crops as adipose, which alone explains its zero G-mean. The leave-one-source-out extension was registered in the protocol and its ten folds were fixed in `loso_manifest.json` before training.

A one-epoch pilot on fold `loso_06` (PolyGabor, ImageNet ResNet and ImageNet YOLO) finished and passed the audit before the campaign, in a separate directory that was later removed. The report was also exercised with synthetic predictions on the ten folds.

The campaign ended on 2026-09-14 with 100 completed runs and no failures: seed 42 in 3 h 17 min and seed 43 in about 3 h 13 min. The audit checked 278 valid runs and the final validation passed. The `yolo_dataset` copies of the YOLO LOSO runs were removed after each completion, as stated in the protocol.

On the macro-F1 of the 5,000 pooled predictions, averaged over seeds 42 and 43, PolyGabor obtained 0.5539, random ResNet-18 0.7001, ImageNet ResNet-18 0.7994, random YOLO11n 0.7532 and ImageNet YOLO11n 0.7893. The drops relative to the random split were 0.239, 0.208, 0.157, 0.109 and 0.162. The original random split overestimated every method, and PolyGabor was the most affected. PolyGabor's relative advantage on `source_a` was not reproduced.

PolyGabor's empty recall was 0.05: when source 06 was tested, training had 31 empty examples, all from source 10, and almost all of the 590 crops were assigned to adipose. Without the empty crops, its macro-F1 rises to 0.6638, still below the CNNs (0.7330 to 0.7894). PolyGabor had higher recall than all CNNs on debris (0.72) and stroma (0.69), and the highest macro-F1 on source 08 (0.662). In the per-source bootstrap, seven of the eight seed/CNN comparisons had an interval entirely in favor of the CNN; the exception was random YOLO11n on seed 42 (−0.005 to 0.256).

## 2026-09-14 — CNN post-training quantization

The first attempt to add litert-torch, ai-edge-litert and ai-edge-quantizer as an extra of `cnn/` would have downgraded typing-extensions, anyio and immutabledict in the campaign's universal lock. The change was reverted, and quantization got the separate uv project `comparison/quantization`, with CPU torch 2.11, because the torchao required by litert-torch does not import with torch 2.7.1.

In the pilots, converting the Keras 3 ResNet through a concrete function produced a 67 kB file without weights, with 13% agreement and NaN outputs; exporting through SavedModel solved it, with 44.7 MB and full agreement. The YOLO converted by litert-torch agreed fully with the saved predictions. Inference preprocessing now uses OpenCV and Pillow, without TensorFlow or PyTorch, and the float32 variant kept 99.6% to 100% agreement across the test conditions for YOLO seed 42.

The first full run was killed by the system for lack of memory after four of the eight models, during calibration of ImageNet ResNet seed 43. The script kept the nine preprocessed test sets and all runtimes in the same process. Evaluation now preprocesses one condition at a time, and each model and seed runs in its own process; the four completed models were reused.

Even so, two more attempts were killed. The kernel did not record an OOM: they were terminated by the session's background task manager because free system memory was low. The export now runs in a subprocess separate from evaluation, and a driver detached from the session, like the one used for the LOSO campaign, completed the eight models, the benchmark and the report. The float32 variant agreed 100% with the original predictions on the clean test and at least 99.6% on the perturbations; the benchmark predictions matched those of the evaluation exactly.

In LiteRT, ResNet-18 went from 44.7 MB in float32 to 22.4 MB with float16 weights and 11.3 MB in int8; YOLO11n went from 6.27 MB to 3.23 MB and 1.73 to 1.81 MB. With one exception, the clean-test macro-F1 change stayed between −0.007 and +0.003. The exception was ImageNet ResNet-18 in static int8, which dropped from 0.970 to 0.862 on seed 42 and from 0.942 to 0.887 on seed 43, with complex recall falling to 0.32 and 0.44. Random ResNet, with the same procedure, lost no performance.

In the single-thread seed 42 benchmark, ImageNet YOLO11n in static int8 took 0.83 ms per image, with a peak RSS of 109 MB, against 3.3 ms and 825 MB for the original PyTorch model and 19 ms and 280 MB for PolyGabor. ResNet-18 in int8 took 4.0 to 4.5 ms, with about 130 MB. Since the peak of the LiteRT variants comes from `ru_maxrss` and that of the original rows is sampled, the comparison favors the original rows. Against these variants, PolyGabor keeps only the smallest file (0.65 MB versus 1.73 MB for int8 YOLO11n), with macro-F1 of 0.79 versus 0.95.
