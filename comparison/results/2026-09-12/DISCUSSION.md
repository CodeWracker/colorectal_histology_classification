# Discussion of the results

178 training runs with evaluation were completed, covering nine pipelines, seven reduced training sizes, full training, class imbalance, label noise and two source-based splits. The full models additionally received eight test perturbations. The campaign includes paired initialization controls: random versus ImageNet ResNet-18 and random versus ImageNet YOLO11n. 22 warm inference benchmarks and 22 cold processes were run.

The best clean-test mean was ImageNet ResNet-18, macro-F1 0.9561, followed by ImageNet YOLO11n, 0.9510. ImageNet YOLO showed the best practical balance between classification, size and latency: 3.204 MB and 3.32 ms on CPU with one thread. PolyGabor kept the advantages of a small package, 0.651 MB, lower inference RAM, faster startup and a more auditable mathematical operation, but it stayed below the pretrained CNNs in aggregate classification and in tumor localization.

## Where to find the evidence

The [generated report](REPORT.md) contains all per-seed tables. [metrics.csv](metrics.csv) gathers training, validation, test and perturbation metrics; [initialization_comparison.csv](initialization_comparison.csv) compares the random and ImageNet configurations; [run_index.csv](run_index.csv) points to each run; [stage_resources.csv](stage_resources.csv) gathers per-stage times and resources; [edge.csv](edge.csv) contains inference; and the [localization report](localization/README.md) details the mosaics. The [protocol](../../PROTOCOL.md), the [work log](../../WORK_LOG.md) and the [image](../../dataset_manifest.json) and [source](../../source_manifest.json) manifests record decisions and preparation.

Each directory in `comparison/runs/2026-09-12` keeps configuration, sample selection, log, timings, resources, model, individual predictions, metrics and figures. Pilots were kept in a separate campaign and do not enter these tables. Large models and data are preserved locally and ignored by Git.

## Performance with full training

Values are means of seeds 42 and 43 on the 500-image random test set. G-mean is computed for each run and then aggregated.

| Pipeline | Macro-F1 | G-mean | Training, s | Model, MB |
| --- | --- | --- | --- | --- |
| PolyGabor, 1 region | 0.7928 | 0.7774 | 20.57 | 0.651 |
| PolyGabor, 1 region + augmentation | 0.7791 | 0.7597 | 78.66 | 0.650 |
| PolyGabor, 9 regions | 0.5225 | 0.4078 | 29.16 | 0.649 |
| PolyGabor, 9 regions + augmentation | 0.5722 | 0.4804 | 98.59 | 0.649 |
| PolyGabor, 25 regions | 0.4269 | 0.0000 | 36.65 | 0.648 |
| ResNet-18 random | 0.9078 | 0.9006 | 173.85 | 45.009 |
| ResNet-18 ImageNet | 0.9561 | 0.9556 | 174.76 | 45.012 |
| YOLO11n random | 0.8618 | 0.8532 | 266.89 | 3.204 |
| YOLO11n ImageNet | 0.9510 | 0.9504 | 269.81 | 3.204 |

MB uses 1,000,000 bytes. Training adds up model construction, preparation, extraction when applicable and fitting. The artifact size does not include libraries and does not represent runtime RAM. The size difference between initializations of the same architecture is negligible, as expected: the initial weights change values, not the number of parameters.

ImageNet ResNet had the highest clean mean, but it ranged from 0.9419 to 0.9703 across the two seeds. ImageNet YOLO ranged from 0.9396 to 0.9624. With only two seeds and ten sources recovered from the file names, the mean difference of 0.0051 between the two does not support a universal ranking. The paired per-image and per-source comparisons are in [paired_tests.csv](paired_tests.csv).

Calibration uses a temperature chosen on validation only and does not change the predicted class. The original mean ECE was 0.2859 for PolyGabor, 0.0269 for random ResNet, 0.0153 for ImageNet ResNet, 0.0436 for random YOLO and 0.0191 for ImageNet YOLO. PolyGabor similarity is a heuristic score; it should not be interpreted as a clinical probability, even after calibration.

## Effect of ImageNet weights

The within-family contrast shows a large advantage associated with the ImageNet configuration when data are scarce. For ResNet, this configuration combines ImageNet weights with the matching input normalization, while the original uses a 0–1 scale; the experiment does not causally attribute the gain to the weights alone. Mean macro-F1 rose from 0.0414 to 0.2842 with one image per class, from 0.1667 to 0.5537 with ten, from 0.5679 to 0.8973 with fifty and from 0.9078 to 0.9561 with full training. With one hundred images per class, the mean changed only from 0.8684 to 0.8711 and the ranges across seeds overlap substantially.

For YOLO, ImageNet raised mean macro-F1 from 0.0290 to 0.5392 with one image per class, from 0.2778 to 0.8342 with ten and from 0.8618 to 0.9510 with full training. The gain stayed positive in every aggregated scenario measured for this family. Under class imbalance, random YOLO had macro-F1 0.7124 and zero G-mean because it completely lost at least one class; the pretrained one reached 0.9046 and G-mean 0.8987.

Pretraining did not improve everything. On source_a, ImageNet ResNet went from macro-F1 0.5407 down to 0.4841 and from G-mean 0.4063 down to zero; on source_b it went up from 0.3444 to 0.6084. With 20 images per class, ImageNet ResNet ranged from 0.2768 to 0.8519 across seeds. Transfer helped a lot on average, but it remained sensitive to the subset and to source shift.

## Where PolyGabor did better

PolyGabor was competitive when the comparator also started from scratch. With two images per class, PolyGabor with augmentation reached a mean macro-F1 of 0.4162, against 0.0721 for random ResNet, 0.0290 for random YOLO and 0.3517 for ImageNet ResNet. With one image per class, this variant reached 0.3709, above both random CNNs and the 0.2842 mean of ImageNet ResNet; ImageNet YOLO was still better, with 0.5392.

With ten images per class, PolyGabor with augmentation obtained 0.5429, practically equal to the 0.5537 mean of ImageNet ResNet and above both random CNNs. With twenty, it reached 0.6213, above the ImageNet ResNet mean of 0.5644, although the wide range of the latter prevents a strong conclusion. ImageNet YOLO stayed ahead at all of these points.

On the source_a split, PolyGabor obtained macro-F1 0.7544, outperforming both ResNets, 0.5407 random and 0.4841 ImageNet, and random YOLO, 0.5637. ImageNet YOLO reached 0.8295. PolyGabor's clearest and most consistent advantage was operational: smallest model, lowest inference RAM, shortest cold start and training that is feasible on CPU alone.

The error gallery contains cases in which only one method is correct. It was rebuilt with the five main pipelines and shows complementarity, without using hand-picked examples to claim aggregate superiority.

## Very few examples and G-mean

![Macro-F1 as a function of training size](learning_curve.png)

![G-mean as a function of training size](learning_curve_gmean.png)

The plots were split into three panels: original methods, CNN initialization control and PolyGabor ablations. This reduces the overlap of nine curves. The subsets are nested within each seed, and the x-axis counts original images per class. Augmentations and regions do not inflate this budget. The band shows the minimum and maximum of two seeds; it is not a confidence interval. Validation keeps 500 labels, so the experiment reduces only the training set.

The original PolyGabor failed with one image per class on both seeds because the polynomial fit does not support a single vector per class. The variants with augmentation or multiple regions could be fitted because they generate more computational vectors, but they still start from a single biological image. These failures are preserved and were not converted to zero F1.

G-mean made visible class collapses hidden by macro-F1. Random ResNet had zero G-mean up to twenty images per class. ImageNet ResNet also had zero G-mean with one and two, despite a positive macro-F1. PolyGabor with 25 regions had zero G-mean even with full training. The [G-mean versus macro-F1 explanation](../../GMEAN_VS_MACRO_F1.md) details the difference: macro-F1 averages per-class precision and recall, whereas G-mean becomes zero when any recall is zero.

## Augmentation and spatial downsampling in PolyGabor

The variants use 1×1, 3×3 and 5×5 grids, corresponding to 1, 9 and 25 regions per image. The bank still has eight pairs of Gabor kernels. Augmentation adds the original, horizontal and vertical flips and brightness ±0.1, in training only.

The cap of 350 vectors per class was kept. With full training, the grids generate up to 4,000, 36,000 or 100,000 candidates, but only 2,800 enter the fit. On seed 42, they represent 2,800 original images in the default setting, 2,170 with augmentation, 2,065 with nine regions, 2,048 with 25 and 2,040 with nine regions plus augmentation. The ablation therefore changes spatial scale and the diversity of retained originals at the same time.

More regions worsened the clean test: mean macro-F1 of 0.7928 with one region, 0.5225 with nine and 0.4269 with 25. Training macro-F1 was also low, indicating a representation/aggregation difficulty beyond overfitting. One plausible cause is that a small region inherits the global label of the image even when its local content does not represent the class; another is the loss of original-image diversity caused by the fixed cap.

Augmentation helped at several low-data points, but not with full training: the full mean macro-F1 dropped from 0.7928 to 0.7791. Training cost rose from 20.57 to 78.66 s. Inference uses only the original image, so size and latency stayed close. The comparison between voting and similarity ranking is in [aggregation_comparison.csv](aggregation_comparison.csv); retrospectively changing the rule did not consistently fix the larger grids.

## Robustness and adverse scenarios

![Perturbations applied to the same tumor image](corruption_gallery.png)

![Robustness comparison](robustness.png)

ImageNet ResNet had a mean macro-F1 of 0.9421 under noise, 0.9138 under JPEG and 0.9312 under rotation, but dropped to 0.6110 under blur, 0.5999 under color shift and 0.5958 under resolution reduction. Random ResNet was better than the pretrained one under blur, 0.7538, and occlusion, 0.8897.

ImageNet YOLO was strong under color shift, 0.9252, occlusion, 0.9263, and rotation, 0.9401. Random YOLO showed two unexpected advantages over the pretrained one: blur, 0.8503 versus 0.6509, and low resolution, 0.8370 versus 0.6353. This does not make up for its clean-test loss, but it shows that the relation between pretraining and robustness depends on the transformation.

Under the fixed imbalance, macro-F1 was 0.4082 for PolyGabor, 0.5515 for PolyGabor with augmentation, 0.7247 for random ResNet, 0.9044 for ImageNet ResNet, 0.7124 for random YOLO and 0.9046 for ImageNet YOLO. With 20% of the training labels corrupted, the values were 0.8248, 0.8831, 0.8881, 0.8581 and 0.9369 for PolyGabor, random ResNet, ImageNet ResNet, random YOLO and ImageNet YOLO, respectively. These adverse scenarios have only seed 42 and one severity; they serve as probes, not population estimates.

## Generalization by source

The codes of the ten sources were recovered from the file names through SHA-256 of the pixels. Source_a tests sources 09/10; source_b tests 06/09; both validate on 08. The original article describes ten slides, but clinical patient identity was not confirmed. Empty exists only in sources 06 and 10, and validation 08 contains no adipose/empty; this limits checkpoint selection.

| Pipeline | Macro-F1 source_a | G-mean source_a | Macro-F1 source_b | G-mean source_b |
| --- | --- | --- | --- | --- |
| PolyGabor | 0.7544 | 0.7501 | 0.5000 | 0.0000 |
| PolyGabor + augmentation | 0.7330 | 0.7031 | 0.5692 | 0.6112 |
| ResNet-18 random | 0.5407 | 0.4063 | 0.3444 | 0.0000 |
| ResNet-18 ImageNet | 0.4841 | 0.0000 | 0.6084 | 0.4619 |
| YOLO11n random | 0.5637 | 0.4467 | 0.5063 | 0.0000 |
| YOLO11n ImageNet | 0.8295 | 0.8102 | 0.8191 | 0.8282 |

![Training and test examples from source_b](source_shift_gallery.png)

Source_a has 580 test images and source_b has 1,403, with different compositions. The gallery shows visual differences and the predictions of the five main pipelines. These splits are more demanding than the random one, but they neither replace an external dataset nor allow attributing every difference exclusively to patient, scanner or slide.

![Complementary hits and errors](error_gallery.png)

The error gallery automatically selects the first cases in which all methods are correct, all are wrong, and each of the five main methods is the only one correct or the only one wrong. The corresponding CSV records index, true class and all predictions.

## Explainability

PolyGabor exposes filters, energy, features, per-class distance and region voting. This link to the decision is more direct for computational auditing. Its dense maps use local distance and may highlight texture or color that does not coincide with the global classification mechanism.

ResNet allows exact CAM because it ends in global average pooling and a linear head. The native resolution is 4×4 for a 128×128 input, so upscaling produces broad regions, not cellular localization. The pipeline now generates and inspects CAM for both random and ImageNet weights.

YOLO uses a 5×5 occlusion explanation: each region is replaced by the mean color and the score drop is measured. Some maps have negative values, indicating that the score increased after hiding the area, and some saturated scores yield nearly zero. This technique was run for ImageNet and random YOLO, but it did not enter the quantitative mosaic test because it is not directly equivalent to CAM or to the PolyGabor dense map.

Being able to inspect a formula or a map easily does not demonstrate histopathological fidelity. There are no internal tumor masks and no clinical evaluation by a specialist. The explanations should be read as computational behavior of the classifiers.

## Quantitative tumor localization

![Localization metrics](localization/localization_metrics.png)

20 validation and 30 test mosaics were used, all 4×4 with two tumor patches and fourteen non-tumor patches. Each patch is explained in isolation at the model's normal resolution and the maps are reassembled only afterwards, with no receptive field or interpolation crossing borders. The ground truth is the known label of each patch.

PolyGabor obtained a per-patch explanation AUROC of 0.8274/0.8233 and top-2 recall of 46.7%/45.0%. Random ResNet obtained 0.9703/0.9860 and 78.3%/85.0%. ImageNet ResNet reached 0.9980/0.9964 and 98.3%/96.7%. The ImageNet configuration also raised the weakly annotated region Dice from 0.6804/0.7077 to 0.8473/0.8441.

The isolated classification score was better than the explanation map for every method: AUROC 0.9638/0.9641 for PolyGabor, 0.9981/0.9984 for random ResNet and 0.9997/0.9996 for ImageNet ResNet. The difference shows that getting a patch's class right does not guarantee that the spatial explanation concentrates evidence on it.

![Classification and explanation: PolyGabor versus random ResNet](localization/localization_examples.png)

The first column shows the mosaic with all known labels and tumor in green. For each model there are two different columns: "classification" colors each patch with its tumor score and writes the value; "explanation" shows the internal spatial map. Dashed yellow is that panel's top-2, and cyan is the explanation threshold chosen on validation only. This layout replaces the old, redundant second column with the known mask.

![Effect of the ImageNet configuration on CAM](localization/localization_pretraining_examples.png)

The paired figure shows that both ResNets classify the patches of the three examples very well, while the ImageNet CAM concentrates the top-2 on the tumors more consistently. Over the 30 full mosaics, the gain of ImageNet over random ResNet in per-mosaic explanation AUROC had bootstrap intervals entirely above zero on both seeds; top-2 recall increased by 0.20 and 0.1167.

The pixel metrics are called weakly annotated region metrics because the whole tumor patch is marked positive without an internal contour. They do not measure real tumor segmentation. The local dataset does not contain `colorectal_histology_large`, so the qualitative extension on the ten large images could not be run.

PolyGabor took about 826.4/813.6 s for the 800 maps and scores of each seed on CPU. The ResNets took 5.3/2.1 s on GPU, with both random and ImageNet initialization. These times reflect the current implementations and devices.

## Training time, CPU, GPU and memory

The hardware was an Intel Core Ultra 9 185H, approximately 32 GB of RAM and an 8 GB NVIDIA RTX 4070 Laptop GPU. CPU-% uses 100% per core; GPU-% is device-wide and includes the desktop; VRAM is queried per PID.

On full seed 42, fitting took 141.38 s for random ResNet, 208.72 s for ImageNet ResNet, 290.11 s for ImageNet YOLO and 240.82 s for random YOLO. The total means across seeds were close within each architecture because of different numbers of epochs: 173.85/174.76 s for the ResNets and 269.81/266.89 s for the YOLOs. Default PolyGabor trained in 20.57 s on average; augmentation increased this to 78.66 s.

Over the whole process, the mean peak RAM was 1.232 GB for PolyGabor, 3.845 GB for random ResNet, 4.264 GB for ImageNet ResNet, 3.480 GB for ImageNet YOLO and 3.471 GB for random YOLO. Mean VRAM was 0, 2.976 GB, 4.022 GB, 0.753 GB and 0.778 GB, respectively. The higher peak of ImageNet ResNet includes importing and converting the Torchvision checkpoint, training, evaluation and figures; it does not imply a larger architecture.

ResNet used approximately 0.6–0.7 CPU cores during training and 87–88% device-wide GPU. YOLO used approximately 3.5 cores and 39–43% device-wide GPU. PolyGabor did not run CUDA; the device-wide GPU activity observed during its process belongs to the rest of the system. The 200 ms sampling can miss short peaks, and energy in joules was not measured.

## Inference and edge computing

![Inference under resource constraints](edge.png)

`cpu1` restricts affinity and libraries to one thread; `cpu4` uses four; `gpu` uses the local RTX. Each benchmark opens a fresh process, loads the model and warms up before 100 batch-1 latencies. Cold start includes process, imports, loading and first prediction; images are already in RAM.

| Model, seed 42 | CPU1 median, ms | CPU4 median, ms | GPU median, ms | CPU1 peak RAM, MB | CPU1 cold, s | Model, MB |
| --- | --- | --- | --- | --- | --- | --- |
| PolyGabor | 19.03 | 19.48 | — | 280.14 | 1.38 | 0.651 |
| ResNet-18 random | 20.19 | 8.20 | 2.48 | 939.35 | 3.64 | 45.009 |
| ResNet-18 ImageNet | 20.73 | 9.55 | 3.08 | 945.87 | 4.00 | 45.012 |
| YOLO11n random | 3.60 | 2.89 | 2.65 | 853.29 | 3.30 | 3.204 |
| YOLO11n ImageNet | 3.32 | 2.88 | 2.49 | 825.00 | 3.22 | 3.204 |

Initialization barely changed latency, RAM or size within a family. YOLO was about six times faster than PolyGabor on CPU1 and offered CPU4 throughput at batch 32 of 814 images/s with ImageNet weights and 888 with random weights. ResNet reached 263/234 images/s on CPU4; PolyGabor, 50.5. On GPU, YOLO was close to 1,630 images/s at batch 32.

The PolyGabor grids increased cost: CPU1 took 27.56 ms with nine regions and 34.12 ms with 25; CPU4 did not help and reached 42.99 ms with 25. Augmentation does not change inference and stayed at 17.55 ms for the package trained with one region.

Twenty-one of the 22 benchmarks reproduced 100% of the saved classes on the first 100 images. ImageNet ResNet on GPU reproduced 99%; the single divergence was an original near tie between complex, 0.4186, and tumor, 0.4169. CPU1 and CPU4 reproduced that run entirely. The artifact records the case instead of treating cross-device execution as bitwise deterministic.

The experiment simulates constraints on x86; it does not emulate ARM, Raspberry Pi, Jetson, RAM limits, energy or prolonged heating. Edge training was not run. PolyGabor is the most plausible candidate for local adaptation without a GPU because of its lower training cost, but YOLO inference is faster on this CPU. The deployment decision needs to be repeated on the target device.

## Usability and program equivalence

The [cnn](../../../cnn/README.md) project offers `dataset`, `train`, `evaluate`, `predict` and `info`, following the structure of PolyGabor. The CLI accepts `--weights auto`, `random`, `imagenet` or an applicable path/configuration. ImageNet ResNet ports 20 convolutions and 20 BatchNorm sets from the official Torchvision checkpoint; the eight-class head is random and all layers are fine-tuned. Random YOLO builds `yolo11n-cls.yaml` and forces `pretrained=False`.

The shareable parameters keep equivalent names; specific arguments are rejected when they do not apply. TensorFlow/Keras and Ultralytics keep their own pipelines, so architecture, augmentations and optimizers differ. The isolated environment pins the versions of TensorFlow, PyTorch, Ultralytics and CUDA.

The eight existing real CLI smoke tests passed, including `info`, `predict` and `evaluate` for both original architectures. The transfer of the 20 ResNet convolutions/BatchNorm sets and its ImageNet roundtrip have a direct regression test; random YOLO was verified through the `pretrained=False` record, the 20 runs of the variant and loading in the three edge modes.

## Integrity, recovery and limits

An accidental move of the folder interrupted two old YOLO runs, label_noise and source_b. They were preserved in `runs/interrupted` and repeated successfully. The final audit checked labels, order, probabilities, accuracy consistency, required files and model size in 178 runs; all passed. There are two known failures of the original PolyGabor in few1. The 180 training events consumed approximately 4.45 hours of summed process time, not counting pilots, localization, edge and preparation.

The final suite passed with 22 tests. The 22 warm benchmarks and the 22 cold processes finished with return code zero. The [validation summary](validation_summary.json) records integrity, tests, CLI commands, benchmark return codes and prediction agreement.

The main limits are two seeds in the main scenarios, one seed in the adverse and source-based ones, a large validation set in few-shot training, few sources, two classes absent from the grouped validation, single-severity perturbations, a fixed PolyGabor vector cap, no external dataset and no internal histopathological masks. The test classifies crops into eight classes; it does not demonstrate per-patient cancer diagnosis.
