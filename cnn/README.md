# CNN: classificação de histopatologia colorretal

Conversão de `../cnn_test.ipynb`, com ResNet-18 e YOLO11n-cls em inicializações aleatória e ImageNet. Estrutura e comandos equivalentes a `../polygarbor`; o carregamento TFDS, splits, console e formatos básicos de avaliação são reutilizados diretamente desse pacote.

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

`uv sync` instala a versão CPU; o extra `gpu` instala as bibliotecas CUDA de TensorFlow, `yolo` habilita YOLO e `pretrained` habilita a importação dos pesos ResNet via Torchvision. O projeto irmão PolyGabor é uma dependência local declarada no uv. `--weights auto` mantém ResNet aleatória e YOLO com `yolo11n-cls.pt`; `--weights imagenet` inicializa o backbone ResNet com `ResNet18_Weights.IMAGENET1K_V1`; `--weights random` constrói a YOLO a partir de `yolo11n-cls.yaml`.

Parâmetros compartilhados: `--data-dir`, `--name`, `--model-dir`, `--out-dir`, `--limit`, `--eval-limit`, `--seed`, `--skip-eval`, `--figures`, `--split`, `--true-label`, `--no-viz`, `--show`, `--quiet`, `--no-color`. Parâmetros CNN: `--architecture`, `--epochs`, `--patience`, `--batch-size`, `--image-size`, `--learning-rate`, `--device`, `--threads`, `--no-augment`, `--weights`. Os parâmetros de taxa de aprendizado e augmentation pertencem à ResNet; YOLO mantém a política do Ultralytics utilizada pelo notebook. `--skip-eval` desativa o relatório final, mas a validação permanece necessária para seleção do checkpoint.

A ResNet mantém a arquitetura e o protocolo do notebook: entrada 128×128, 8 blocos residuais, Adam 1e-3, batch 32, máximo de 100 épocas, flips e brilho, early stopping por acurácia de validação (paciência 8) e redução de LR por loss de validação (paciência 4). Na variante ImageNet, os 20 kernels convolucionais e 20 conjuntos de BatchNorm do checkpoint oficial Torchvision são portados para a mesma rede TensorFlow, a cabeça de oito classes começa aleatória e a entrada usa a normalização ImageNet. Todas as camadas continuam ajustáveis. A ordem de leitura é determinística e há shuffle apenas no treino. A YOLO usa a mesma arquitetura e política Ultralytics nas duas inicializações; somente a origem dos pesos muda.

`src/cnn/` contém `classifier.py`, `dataset.py`, `evaluation.py`, `visualize.py`, `console.py` e `cli.py`. `tests/` verifica arquitetura, treino, persistência, ordenação das probabilidades e métricas. `../comparison/` contém o protocolo, executor dos cenários, monitoramento e relatório dos experimentos.

Cada modelo salvo contém `model.json` e `model.keras` (ResNet, somente estado de inferência, sem momentos do Adam) ou `model.pt` (YOLO). O histórico fica no diretório de treino em CSV/JSON e há um registro incremental por época da ResNet. `predict` gera ranking e explicação: CAM nativo da ResNet com sobreposição ampliada ou sensibilidade à oclusão para YOLO. Esses mapas são evidência do modelo, não máscaras de segmentação.

Se o TensorFlow não localizar as bibliotecas CUDA instaladas pelo pip, execute `uv run python benchmarks/setup_cuda.py` dentro de `cnn`; o script cria os links locais descritos na [documentação oficial do TensorFlow](https://www.tensorflow.org/install/pip#linux). Ele não altera o driver nem bibliotecas do sistema. As versões TensorFlow 2.20 e PyTorch 2.7.1 estão fixadas para compartilhar CUDA 12 com o driver usado nos experimentos.
