# Quantitative tumor localization in mosaics

20 validation and 30 test mosaics were evaluated, all 4×4 and 600×600 pixels, with two tumor patches and fourteen non-tumor patches. The 800 patches used are unique within each split. The full models of seeds 42 and 43 were reused without retraining.

![Localization metrics](localization_metrics.png)

| Method | Seed | Explanation AUROC/patch | Explanation AP/patch | Top-2 recall | Both in top-2 | Classifier AUROC/patch | Weak-region AUROC | Weak-region Dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| polygarbor | 42 | 0.8274 | 0.3640 | 0.4667 | 0.1667 | 0.9638 | 0.7702 | 0.4000 |
| polygarbor | 43 | 0.8233 | 0.3543 | 0.4500 | 0.1667 | 0.9641 | 0.7669 | 0.3938 |
| resnet18 | 42 | 0.9703 | 0.7195 | 0.7833 | 0.5667 | 0.9981 | 0.9433 | 0.6804 |
| resnet18 | 43 | 0.9860 | 0.9158 | 0.8500 | 0.7000 | 0.9984 | 0.9457 | 0.7077 |
| resnet18_imagenet | 42 | 0.9980 | 0.9868 | 0.9833 | 0.9667 | 0.9997 | 0.9871 | 0.8473 |
| resnet18_imagenet | 43 | 0.9964 | 0.9806 | 0.9667 | 0.9333 | 0.9996 | 0.9856 | 0.8441 |

The main analysis uses the known label of each patch. AUROC and AP check whether the mean evidence of the map ranks tumor patches above the others. Top-2 recall measures how many of the two tumors appear among the two patches with the highest evidence; both in top-2 requires getting the pair exactly right. Classifier AUROC uses the classifier's tumor score for each isolated patch and makes it possible to tell decision errors apart from explanation-map errors.

![Examples of mosaics, labels and maps](localization_examples.png)

Each patch is processed in isolation and only then are the maps reassembled. Thus no receptive field or interpolation crosses the artificial borders of the mosaic. The figure shows a ground-truth column and, for each method, one column with the tumor classification score and another with the explanation map. Green marks the tumor ground truth, dashed yellow shows each panel's top-2 and cyan shows the explanation threshold selected on validation. PolyGabor values are heuristic similarities and ResNet values are uncalibrated softmax; they are meant for ranking within each method.

![CAM comparison by configuration](localization_pretraining_examples.png)

The comparison keeps architecture, images and layouts. The ImageNet configuration changes the initial weights and applies the normalization those weights require; therefore the contrast does not mathematically isolate the initialization alone. It shows how the classification ranking and the CAM change under the transfer configuration used.

The PolyGabor score is the negative logarithmic distance to tumor on a dense 75×75 grid per patch. The ResNet CAM is computed on its trained 128×128 input, before the softmax, from the 4×4 spatial activations and the tumor class weights, with ReLU. Each map is interpolated only within its own 150×150 patch.

The threshold of each method and seed maximizes Dice exclusively on validation. The pixel metrics were kept as a secondary analysis of weakly annotated regions: the whole area of a tumor patch is positive because there is no histopathological contour inside it. They do not measure real cell or tumor segmentation. The random AUROC baseline is 0.5 and the positive prevalence is 12.5%. The intervals resample the 30 whole mosaics with 5,000 draws.

Raw artifacts: runs/2026-09-12/localization. The telemetry_manifest.json manifest points to the telemetry of version 2, of the ImageNet extension and of the refreshes. The qualitative part on the large images was not run because the local archive contains the 5,000 patches, without the separate colorectal_histology_large dataset.
