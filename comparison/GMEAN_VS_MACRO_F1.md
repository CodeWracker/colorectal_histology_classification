# How to interpret multiclass G-mean and macro-F1

Both metrics give weight to every class, but they answer different questions. Macro-F1 balances the precision and recall of each class; multiclass G-mean measures how balanced the recalls are and strongly penalizes the least recognized class.

## Definitions

For each class $i$, $Recall_i = TP_i/(TP_i+FN_i)$ and $Precision_i = TP_i/(TP_i+FP_i)$. In this comparison, $C=8$.

$$G\text{-mean} = \left(\prod_{i=1}^{C} Recall_i\right)^{1/C}$$

$$Macro\text{-}F1 = \frac{1}{C}\sum_{i=1}^{C}\frac{2\,Precision_i\,Recall_i}{Precision_i+Recall_i}$$

G-mean is computed as `exp(mean(log(recalls)))` for numerical stability when all recalls are positive. If any recall is zero, it returns exactly zero. We add no epsilon or correction that would hide a completely ignored class. When a class has no real examples in the evaluated set, its recall cannot be estimated and the eight-class G-mean is recorded as `null` instead of inventing a value. This happens in the validation sets of the source splits, which do not contain every class; their test sets contain all eight.

## What each metric reveals

| Property | Macro-F1 | Multiclass G-mean |
| --- | --- | --- |
| Uses precision | Yes, explicitly | Not explicitly; false positives hurt the recall of the classes those errors come from |
| Uses recall | Yes | Yes, exclusively |
| Aggregation | Arithmetic mean of per-class F1 | Geometric mean of recalls |
| A class is never recognized | Contributes zero F1, but the aggregate can remain reasonable | The whole result becomes zero |
| Trade-off between strong and weak classes | Gains on strong classes can offset part of the loss on weak ones | Penalizes imbalance and near-zero recalls more |
| Practical question | Does the model keep precision and coverage balanced per class? | Can the model recognize every class without leaving one behind? |

For example, consider recalls of 0.95 for seven classes and a recall of zero for the eighth. G-mean is zero, even if accuracy is high. Macro-F1, in contrast, can be relatively high, depending on the precisions; it cannot be computed from the recalls alone. If the recall of the eighth class rises to 0.10, G-mean goes up to approximately 0.717. This shows how moving from no correct predictions to a few for one class can produce a jump in the curve. These numbers are a mathematical example, not results from a training run.

## How to read the few-shot curves

The x-axis counts distinct original images used in training, labeled per class and in total. Augmentations and patches do not increase this budget: an image split into nine regions still counts as one original image. The y-axis shows macro-F1 or G-mean on the same test set. The curves use the same nested subsets and seeds; the band shows the minimum and maximum across seeds, not a statistical confidence interval.

A rising macro-F1 together with a zero G-mean indicates that the model improved on some classes but still does not recognize at least one. A rising G-mean with little change in macro-F1 may indicate better coverage of the weakest class, offset by false positives. Good values on both metrics suggest better balance, which should be confirmed with the individual recalls and precisions and with the confusion matrix.

With few examples, G-mean can vary a lot across seeds, especially near zero recall. A curve at zero does not mean that all predictions are wrong. A classifier fitting failure is not a zero G-mean either: in that case there is no evaluated model and the point is missing, with the failure recorded in the index.

## Caveats in this comparison

The scarcity scenarios keep 500 labeled validation examples: they restrict training, not the whole annotation budget. The comparison between random splits and source splits involves different populations and sizes. The `empty` class is concentrated in two sources, which affects both the interpretation of generalization and the possibility of covering every class in three splits without sharing sources.

G-mean does not measure calibration, explanation quality, time, memory or clinical risk. Nor does it prove generalization across patients. The report presents those dimensions separately. The implementation is in `cnn/src/cnn/evaluation.py`, and earlier values were recomputed from the saved predictions, without retraining models.

## Examples measured in this campaign

On seed42, PolyGabor with five original images per class obtained a macro-F1 of approximately 0.431, but zero G-mean. This means that some classes were usefully learned, but at least one received no correct predictions. On the source_b source test, ResNet reached a macro-F1 of approximately 0.344 and also a zero G-mean. The results and the exact per-class recalls are in the corresponding `eval/test_clean/metrics.json`.

These examples justify presenting both curves: macro-F1 tracks gradual gains in precision and recall, while G-mean makes a class's lack of coverage explicit. The values come from specific runs, not means over all configurations.
