# CNN: colorectal histopathology classification

Conversion of `../cnn_test.ipynb`, with ResNet-18 and YOLO11n-cls under random and ImageNet initialization. Structure and commands are equivalent to `../polygarbor`; TFDS loading, splits, console and the basic evaluation formats are reused directly from that package.

```bash
cd cnn
uv sync --extra gpu --extra yolo
uv run cnn dataset --data-dir ../polygarbor/data --preview 9 --export-per-class 1
uv run cnn train --data-dir ../polygarbor/data --model-dir artifacts/resnet --out-dir artifacts/train-resnet --figures --device gpu
uv run cnn train --architecture resnet18 --weights imagenet --data-dir ../polygarbor/data --model-dir artifacts/resnet-imagenet --out-dir artifacts/train-resnet-imagenet --device gpu
uv run cnn train --architecture yolo11n --weights random --data-dir ../polygarbor/data --model-dir artifacts/yolo-random --out-dir artifacts/train-yolo-random --device gpu
uv run cnn evaluate --data-dir ../polygarbor/data --model-dir artifacts/resnet --split test
uv run cnn predict -i ../polygarbor/artifacts/dataset/samples/00_tumor_1.png --model-dir artifacts/resnet
uv run cnn info --model-dir artifacts/resnet
```

`uv sync` installs the CPU version; the `gpu` extra installs the CUDA libraries for TensorFlow, `yolo` enables YOLO and `pretrained` enables importing the ResNet weights through Torchvision. The sibling PolyGabor project is a local dependency declared in uv. `--weights auto` keeps ResNet random and YOLO with `yolo11n-cls.pt`; `--weights imagenet` initializes the ResNet backbone with `ResNet18_Weights.IMAGENET1K_V1`; `--weights random` builds YOLO from `yolo11n-cls.yaml`.

Shared parameters: `--data-dir`, `--name`, `--model-dir`, `--out-dir`, `--limit`, `--eval-limit`, `--seed`, `--skip-eval`, `--figures`, `--split`, `--true-label`, `--no-viz`, `--show`, `--quiet`, `--no-color`. CNN parameters: `--architecture`, `--epochs`, `--patience`, `--batch-size`, `--image-size`, `--learning-rate`, `--device`, `--threads`, `--no-augment`, `--weights`. The learning-rate and augmentation parameters belong to ResNet; YOLO keeps the Ultralytics policy used by the notebook. `--skip-eval` disables the final report, but validation is still required for checkpoint selection.

ResNet keeps the notebook's architecture and protocol: 128×128 input, 8 residual blocks, Adam 1e-3, batch 32, at most 100 epochs, flips and brightness, early stopping on validation accuracy (patience 8) and LR reduction on validation loss (patience 4). In the ImageNet variant, the 20 convolutional kernels and 20 BatchNorm sets of the official Torchvision checkpoint are ported to the same TensorFlow network, the eight-class head starts random and the input uses ImageNet normalization. All layers remain trainable. The reading order is deterministic and shuffling happens only during training. YOLO uses the same architecture and Ultralytics policy under both initializations; only the origin of the weights changes.

`src/cnn/` contains `classifier.py`, `dataset.py`, `evaluation.py`, `visualize.py`, `console.py` and `cli.py`. `tests/` checks architecture, training, persistence, probability ordering and metrics. `../comparison/` contains the protocol, the scenario runner, monitoring and the experiment report.

Each saved model contains `model.json` and either `model.keras` (ResNet, inference state only, without Adam moments) or `model.pt` (YOLO). The history is stored in the training directory as CSV/JSON, and ResNet also writes an incremental per-epoch log. `predict` produces a ranking and an explanation: native ResNet CAM with an upscaled overlay, or occlusion sensitivity for YOLO. These maps are model evidence, not segmentation masks.

If TensorFlow cannot find the CUDA libraries installed by pip, run `uv run python benchmarks/setup_cuda.py` inside `cnn`; the script creates the local links described in the [official TensorFlow documentation](https://www.tensorflow.org/install/pip#linux). It does not change the driver or system libraries. TensorFlow 2.20 and PyTorch 2.7.1 are pinned so that they share CUDA 12 with the driver used in the experiments.
