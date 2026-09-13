# Classificação de histopatologia colorretal

Dois projetos com comandos equivalentes de dataset, treino, avaliação, predição e inspeção: [PolyGabor](polygarbor/README.md), baseado em descritores de textura/cor e subespaços polinomiais, e [CNN](cnn/README.md), com ResNet-18 e YOLO11n convertidas do notebook e controles de inicialização aleatória/ImageNet.

A [comparação experimental](comparison/README.md) registra splits compartilhados, generalização por origem, poucos exemplos, augmentations, grades de patches, condições adversas, localização quantitativa em mosaicos, CPU/GPU, memória, tamanho de modelo e inferência com restrição de recursos. Consulte a [discussão e avaliação visual](comparison/results/2026-09-12/DISCUSSION.md), o [relatório](comparison/results/2026-09-12/REPORT.md), o [teste de localização](comparison/results/2026-09-12/localization/README.md), as [curvas e tabelas](comparison/results/2026-09-12/) e a [explicação de G-mean versus macro-F1](comparison/GMEAN_VS_MACRO_F1.md).

Os notebooks originais foram preservados. Modelos e dados grandes ficam no disco e são ignorados pelo Git; configurações, logs, métricas e figuras permitem auditar cada execução.
