# Protocol registered before the experiments

Compare default PolyGabor (1 patch, 3 levels, at most 350 vectors/class), the notebook's ResNet-18 (from scratch) and YOLO11n-cls (ImageNet transfer). Preserve all existing data and artifacts.

- TFDS colorectal_histology 2.0.0, deterministic order, original 4000/500/500 splits. Same examples and labels for all methods.
- Save indices, SHA-256 hashes of the images and per-class counts. Check for exact duplicates across splits. TFDS does not provide patient/slide: internal test on crops, without claiming cross-patient generalization.
- Full training, 1, 2, 5, 10, 20, 50 and 100 examples/class (nested subsets), class imbalance (10% of the tumor/complex/stroma/mucosa classes), uniform label noise on 20% of training; seeds 42 and 43 in the scarcity scenarios to measure sensitivity to sampling/initialization.
- ResNet: 128 px, batch 32, Adam 1e-3, up to 100 epochs, early stopping on val_accuracy/patience 8 and LR reduction on val_loss/patience 4. Horizontal/vertical flips and brightness ±0.1 as in the notebook.
- YOLO: 128 px, batch 32, up to 100 epochs and patience 8; keep the backend's own transformations and document the differences.
- Selection exclusively on validation. No hyperparameter tuning on the test set.
- Full clean test and fixed perturbations: blur, noise, brightness, color shift, JPEG compression, low resolution, occlusion and rotation. Same perturbed pixels for all methods. They are not equivalent to another clinical dataset.
- Accuracy, macro-F1, balanced accuracy, MCC, kappa, per-class precision/recall/F1, confusion matrix, top-2, OVR AUROC, NLL, Brier and ECE. PolyGabor normalized similarity is a heuristic score, not a calibrated probability; voting confidence is not used as a probability.
- Paired bootstrap on the test set for the accuracy difference and exact McNemar; crop-level CIs do not measure uncertainty across patients; exploratory tests.
- Separate processes per run. Stages: loading, preparation/extraction, construction, fitting, saving, reloading, evaluation and explanations. CPU in core-% (100% = one core), sampled RSS RAM, CPU-seconds, device-wide GPU-% and per-PID VRAM. Device-wide GPU includes the desktop; measured memory is not consumption exclusive to the model.
- Warm batch-1 inference (median/p95 latency), batch throughput, cold load in a subprocess, CPU and GPU for the CNNs. MB = 1,000,000 bytes; report the inference package separately from the checkpoint with optimizer state.
- Figures: curves, matrices, robustness, trade-offs, agreeing and disagreeing cases, ResNet CAM and PolyGabor maps/descriptors. Maps are neither segmentation nor proof of clinical validity.

Each run will have config, status, log, timings, resources, model, history, individual predictions and figures. Failures and pilots will be recorded and excluded from the main means. Results will be consolidated in CSV and Markdown.

Extension requested before the main experiments: macro-F1 versus examples per class curve (nested subsets, seeds 42/43). Edge will be a simulation on an x86 CPU with affinity restricted to one logical processor and one thread, batch 1, compared with a CPU with four threads and a GPU. It does not represent measurements on Raspberry Pi, ARM or Jetson. Memory will not be artificially limited; the peaks will indicate whether the models would fit hypothetical budgets. Edge training will be discussed based on the measured cost, without extrapolating GPU time to ARM.

## Source generalization extension

A later inspection of the original files revealed ten CRC-Prim-HE codes in the crop names. They will be recovered by exact SHA-256 matching of the pixels, without changing the main campaign splits. Two new splits were fixed before training the grouped models: source_a trains on 01–07, validates on 08 and tests on 09/10; source_b trains on 01–05/07/10, validates on 08 and tests on 06/09. Each will have the three methods and seed42, without hyperparameter tuning on its test set. The results will not be mixed into the random-split means.

The empty class occurs only in sources 06 and 10, which makes it impossible to cover all eight classes in training, validation and test at the same time without sharing sources. The chosen splits keep all classes in training and test; validation 08 contains no adipose/empty. This restriction and the difference in test size and composition will be reported. The codes indicate a source/slide inferred from the name, not a confirmed clinical patient identity. The article describes ten independent slides: https://pmc.ncbi.nlm.nih.gov/articles/PMC4910082/.

## Leave-one-source-out extension

A review of `source_a` and `source_b` found three limitations. First, they were two manually chosen partitions, with one seed each. Second, `source_b` concentrates 590 of its 1,403 test crops in the empty class, with only 35 examples of that class in training, all from source 10. Third, validation on source 08 contains neither adipose nor empty, while the CNNs use early stopping on val_accuracy. These scenarios become a historical record and do not support comparison between methods.

Before any new grouped training, `loso.py` fixed ten leave-one-source-out folds in `loso_manifest.json`. Each fold tests one entire source. Validation is 10% of each class from the remaining nine sources, with a minimum of one crop and partition seed 20260913. Training uses the rest. The indices are the same for all methods and seeds. polygarbor, resnet18, resnet18_imagenet, yolo11n_random and yolo11n will be run with the hyperparameters, stopping criteria and cap of 350 vectors/class of the main campaign, on seeds 42 and 43. No hyperparameter will be tuned on the test sets.

The primary metric is macro-F1 over the 5,000 pooled predictions of the ten folds, in which each crop is predicted by a model that did not see its source. G-mean, balanced accuracy, per-class recall and macro-F1 without the empty crops will also be reported. Per-fold macro-F1 on the present classes only and the paired difference between PolyGabor and each CNN, with a bootstrap over the ten sources, will be reported as well. Validation comes from the same sources as training, but the test remains source-disjoint. The data come from ten images from a single institute and scanner, so the result measures generalization across images of this collection, not external validation.

To fit on disk, the `yolo_dataset` image copy of each YOLO LOSO run is removed after completion. It is fully derived from the cache and the manifest indices.

## CNN post-training quantization extension

Before running, a comparison with inference-only quantized versions of the CNNs was fixed, without retraining and without quantization during training. PolyGabor stays at full precision as a reference. The full models of resnet18, resnet18_imagenet, yolo11n_random and yolo11n on seeds 42 and 43 are included.

All variants use LiteRT (TFLite) and the same ai-edge-litert interpreter with XNNPACK. The Keras ResNet is exported as a SavedModel with a fixed batch size of 1 and converted by the TensorFlow 2.20 converter; YOLO is converted from PyTorch with litert-torch. From each float32 model, ai-edge-quantizer generates three variants: float16 weights (float casting), dynamic int8 (per-channel int8 weights and activations quantized at run time) and static int8 (int8 weights and activations, with float32 input and output). Calibration uses 200 crops from the training split, 25 per class with seed 20260914; validation and test are not used.

Inference preprocessing depends on neither TensorFlow nor PyTorch: OpenCV bilinear resizing for ResNet, with ImageNet normalization when applicable, and Pillow bilinear resizing for YOLO, equivalent to the transformation saved in the checkpoint. The float32 variant separates the effect of runtime and preprocessing from the effect of quantization. Its agreement with the original predictions on the clean test must be at least 99%; below that the conversion is considered invalid and the run is stopped.

The variants will be evaluated on the clean test and on the eight fixed perturbations, with the same pixels as the campaign. File size, accuracy, macro-F1, G-mean, balanced accuracy, per-class recall, agreement with the original model and with the float32 variant, and the macro-F1 difference will be reported. Median and p95 batch-1 latency, including preprocessing, and peak RSS will be measured in fresh processes in the cpu1 and cpu4 modes of the edge benchmark, for seed 42, with no training running.

The environment lives in `comparison/quantization`, with its own lock, because litert-torch requires torch 2.11 and would change versions pinned in the campaign environment. The models trained with torch 2.7.1 are loaded unchanged.

## G-mean, augmentation and patch grid extension

At the authors' request, the comparison adds exact multiclass G-mean, without smoothing zero recall, recomputed from the existing predictions. Values without support in every class will be undefined. The plot will count original images, and a separate document will explain the difference from macro-F1.

Four additional PolyGabor variants: 3×3 grid (9 descriptors), 5×5 grid (25 descriptors), 1×1 grid with augmentation and 3×3 grid with augmentation. The bank keeps eight kernel pairs: the number of regions is not the number of filters. Augmentation is applied in training only: the original plus three deterministic variants with horizontal/vertical flips and brightness ±0.1, clipped to the valid RGB range. It is a controlled policy close to that of ResNet, not a copy of YOLO's RandAugment. The variants keep levels=3, cap=350 vectors/class and voting, recording both the extracted vectors and the number of original images actually retained. The curves and scenarios use the same indices/seeds as the other approaches.

The four variants will be run with full training and the seven scarcity sizes on two seeds; imbalance, label noise and the two source splits will use seed42. This makes it possible to separate the effect of the grid from the effect of augmentation without artificially inflating the data-quantity axis.

## Quantitative localization in mosaics

Before running, a synthetic binary localization test of tumor versus the other classes was fixed. 20 validation and 30 test mosaics will be built, each with a 4×4 grid of 150×150-pixel patches, resulting in 600×600 pixels. Each mosaic will have two tumor patches in random positions and fourteen patches from the other classes. Patches will not be reused within a split; validation and test remain disjoint according to the original splits. Layout, indices and labels will be saved.

The full models of seeds 42 and 43 will be reused, without retraining or test-based selection. For PolyGabor, the spatial score will be `-log1p(distance to tumor)` computed by the dense 75×75 descriptor. For ResNet-18, the exact CAM of the global-average-pooling/linear head will be used, before softmax and after ReLU. The network will be extended fully convolutionally to a 600×600 input with the same weights; numerical equivalence at a 128×128 input will be verified. Both maps will be linearly interpolated to 600×600.

The threshold that maximizes pooled Dice will be chosen exclusively on the validation mosaics of each method and seed. On the test set, pixel AUROC and average precision, Dice and IoU at the fixed threshold will be reported, plus per-mosaic means with a 95% bootstrap interval resampling whole mosaics. Neighboring pixels will not be treated as independent replicates for the interval. The positive prevalence is 12.5%; the baseline that marks everything as tumor has Dice 0.2222 and AUROC 0.5.

The comparison measures localization in artificial mosaics made of labeled homogeneous patches. The mask borders come from the assembly, not from pixel-level histopathological annotation. The result does not demonstrate tumor segmentation on real slides. The qualitative part on the ten large images depends on the local availability of the `colorectal_histology_large` dataset and will be kept separate from the quantitative test.

### Methodological revision after visual inspection

The first run revealed that computing the map on the full mosaic let the receptive field and the interpolation cross artificial borders between independent images. This design was replaced by version 2 of the protocol. Each 150×150 patch is now explained in isolation at the method's normal input resolution; the map is resized only within its own patch, and the sixteen maps are then reassembled without mixing at the borders.

The primary evaluation unit becomes the patch, since that is the level of the ground truth provided by the dataset. The explanation score of a patch is the mean of its map. Between-patch AUROC and AP, recovery of the two tumors among the two highest scores, the rate of mosaics with both recovered and per-patch Dice/IoU with a threshold chosen only on validation will be reported. The tumor classification score of each patch will also be evaluated separately, to distinguish classifier error from explanation error.

The pixel metrics are now called weakly annotated region localization and are kept as a secondary analysis. Every pixel of a tumor patch receives a positive label only by inheriting the image's class; therefore these metrics prove neither internal localization nor histological segmentation. The figures will mark the tumor patches in green, the two patches most highlighted by the map in yellow and the threshold contour in cyan, computed separately within each patch.

## CNN initialization control

Before this extension, two controls were added to separate architecture from pretraining: `resnet18_imagenet` and `yolo11n_random`. They will run the same splits, nested subsets, seeds, perturbations, metrics, monitoring and stopping criteria as their corresponding architectures. Full training on seeds 42/43, the seven scarcity scenarios on both seeds, imbalance and label noise on seed 42 and the two source splits on seed 42 will be included.

`resnet18_imagenet` receives the 20 convolutional kernels and 20 BatchNorm sets of `Torchvision ResNet18_Weights.IMAGENET1K_V1`; the eight-class head is random, all layers are trainable and the input uses ImageNet mean/std. The port keeps the Keras architecture used by the project and records the origin in `initialization.json`. Since padding and BatchNorm numerical parameters are not a byte-for-byte reproduction of the PyTorch runtime, the experiment measures initialization with ported ImageNet filters, not exact logit equivalence with Torchvision.

`yolo11n_random` builds `yolo11n-cls.yaml` and calls training with `pretrained=False`. `yolo11n` still starts from `yolo11n-cls.pt`; architecture, input size, augmentation policy, automatic optimizer and all other arguments remain the same. For YOLO, the origin of the weights is the intended controlled factor. For ResNet, the ImageNet configuration also requires ImageNet mean/std, while the original variant uses a 0–1 scale; therefore its contrast combines initialization and compatible normalization and does not causally identify the weights alone. Comparisons between ResNet and YOLO still include differences in framework and training policy.

Localization version 3 adds the CAM of `resnet18_imagenet` on the same mosaics, seeds and metrics as random ResNet. The PolyGabor and random ResNet maps from version 2 will be reused without recomputation; their previous raw artifacts will be preserved with the `_v2` suffix. A dedicated figure will compare, for each ResNet initialization, the per-patch classification scores and the CAM, so that a correct decision is not confused with a correct spatial explanation.
