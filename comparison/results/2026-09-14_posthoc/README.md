# Análises post-hoc de 2026-09-14

Estas análises foram feitas depois da campanha `2026-09-12`, durante a revisão do artigo `ichi-polygarbor`, com autorização explícita dos autores. Não fazem parte do protocolo registrado em `../../PROTOCOL.md` e devem ser descritas no artigo como exploratórias.

## Normalização das distâncias no desbalanceamento

Script: `comparison/posthoc_imbalance_normalization.py`. Saídas: `imbalance_normalization.csv` e `imbalance_normalization.json`.

Nenhum modelo foi retreinado. Os modelos salvos de `imbalance__polygarbor__seed42`, `full__polygarbor__seed42` e `full__polygarbor__seed43` foram carregados e só a regra de decisão mudou:

- `raw`: argmin D_c, a regra registrada. Reproduz o macro-F1 de teste 0,408 do cenário imbalance.
- `insample`: argmin D_c / r_c, com r_c a mediana de D_c nos próprios vetores de ajuste avaliados pelo modelo construído com eles.
- `crossfit`: igual, mas cada vetor é avaliado por um modelo ajustado sem ele (5 folds dentro da classe).
- `count`: argmin N_c · D_c. Dividir a matriz de espalhamento por N_c escala todos os valores singulares e s_min = eps · s_max por 1/N_c, sem mudar bases, projeções ou limiares relativos, então cada nível de D_c é multiplicado por N_c. É o PMD com espalhamento normalizado pelo número de vetores. Com classes balanceadas, as predições são idênticas às de `raw`.

Nenhum rótulo de validação ou de teste foi usado para definir as regras. Resultados de teste:

| modelo | regra | macro-F1 | G-mean |
| --- | --- | --- | --- |
| imbalance seed 42 | raw | 0,408 | 0,000 |
| imbalance seed 42 | insample | 0,434 | 0,000 |
| imbalance seed 42 | count | 0,552 | 0,379 |
| imbalance seed 42 | crossfit | 0,653 | 0,625 |
| full, média seeds 42/43 | raw = count | 0,793 | 0,777 |
| full, média seeds 42/43 | insample | 0,780 | 0,767 |
| full, média seeds 42/43 | crossfit | 0,775 | 0,761 |

As regras foram escolhidas depois de observar a falha e avaliadas com uma seed e uma razão de desbalanceamento.

## Pegada de memória do PolyGabor

Script: `comparison/posthoc_footprint_scaling.py`. Saída: `footprint_scaling.csv`.

Os descritores retidos pelo modelo `full__polygarbor__seed42` (350 por classe) foram subamostrados, com seed 42, para 2, 5, 10, 20, 50, 100 e 350 vetores por classe com 8 classes, e para 2 e 4 classes com 350 vetores. Nenhum descritor foi extraído de novo e nada foi avaliado no teste. `saved_mb` é o tamanho escrito por `save()`, `model_array_mb` são os arrays dos modelos PMD mais as amostras retidas, `fit_peak_mb` e `eval_peak_mb` são picos do `tracemalloc`, e `distance_median_ms` é a mediana de 30 cálculos de distância de um vetor (4 threads). A medição rodou antes de iniciar o treino das CNNs na CPU.

## Treino das CNNs na CPU

Script: `comparison/cpu_training.py`. Execuções: `runs/2026-09-14_cpu`. Usa o mesmo `run.py`, seed 42, 4 threads e `--device cpu`, com 10 imagens por classe e com o treino completo. Durante as execuções, a máquina também rodava o navegador e, por poucos segundos, os scripts acima e a geração de figuras do artigo, então os tempos têm algum ruído.
