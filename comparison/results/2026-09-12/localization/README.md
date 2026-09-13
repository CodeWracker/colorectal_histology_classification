# Localização quantitativa de tumor em mosaicos

Foram avaliados 20 mosaicos de validação e 30 de teste, todos 4×4 e 600×600 pixels, com dois patches de tumor e quatorze não tumorais. Os 800 patches utilizados são únicos dentro de cada split. Os modelos completos das seeds 42 e 43 foram reutilizados sem retreino.

![Métricas de localização](localization_metrics.png)

| Método | Seed | AUROC pixels | AP pixels | Dice | IoU | AUROC médio/mosaico (IC95%) | Dice médio/mosaico (IC95%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| polygarbor | 42 | 0.6408 | 0.1891 | 0.2594 | 0.1490 | 0.6480 (0.5972–0.7009) | 0.2668 (0.2221–0.3160) |
| polygarbor | 43 | 0.6387 | 0.1858 | 0.2588 | 0.1487 | 0.6455 (0.5946–0.6985) | 0.2668 (0.2222–0.3161) |
| resnet18 | 42 | 0.8964 | 0.4523 | 0.5584 | 0.3873 | 0.8988 (0.8763–0.9197) | 0.5742 (0.5304–0.6178) |
| resnet18 | 43 | 0.8977 | 0.5775 | 0.5889 | 0.4174 | 0.8974 (0.8732–0.9180) | 0.5897 (0.5496–0.6276) |

O threshold de cada método e seed maximiza Dice somente na validação. AUROC e average precision não usam threshold. O baseline aleatório de AUROC é 0,5; marcar todos os pixels como tumor produz Dice 0,2222 nesta prevalência. Os intervalos reamostram os 30 mosaicos inteiros por 5.000 draws.

![Exemplos de mosaicos, máscaras e mapas](localization_examples.png)

O escore PolyGabor é `-log1p(distância para tumor)` na grade densa 75×75. O CAM da ResNet é calculado antes do softmax a partir das ativações espaciais e dos pesos da classe tumor, com ReLU. A ResNet foi estendida de forma totalmente convolucional para 600×600; a maior diferença observada em 128×128 fica registrada em raw_metrics.json. Os dois mapas foram interpolados linearmente para a resolução do mosaico.

Esta avaliação mede separação espacial em uma construção sintética cujas regiões de 150×150 já possuem rótulo de classe. Ela não usa contornos celulares ou anotação de tumor dentro de uma lâmina e não valida segmentação clínica. A interpolação, o campo receptivo e as bordas entre patches afetam as métricas em pixels. A comparação pareada por mosaico está em paired_comparison.csv; mapas nativos e layouts auditáveis estão no diretório da execução.

Artefatos brutos: `runs/2026-09-12/localization`. Tempos e recursos estão em timings.json e resources.json. A parte qualitativa nas imagens grandes não foi executada porque o arquivo local contém os 5.000 patches, sem o dataset separado `colorectal_histology_large`.
