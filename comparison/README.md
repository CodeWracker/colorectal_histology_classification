# Experimentos reproduzíveis

A implementação está em `../cnn`, e o método original permanece em `../polygarbor`. O [protocolo](PROTOCOL.md) registra as decisões antes dos experimentos. Comece pela [discussão e avaliação visual](results/2026-09-12/DISCUSSION.md). Os resultados consolidados ficam em [results/2026-09-12/REPORT.md](results/2026-09-12/REPORT.md); cada linha do índice aponta para a respectiva execução.

Executar a partir da raiz do repositório:

```bash
uv sync --project cnn --extra gpu --extra yolo
cnn/.venv/bin/python comparison/prepare.py
cnn/.venv/bin/python comparison/prepare_groups.py
cnn/.venv/bin/python comparison/suite.py --campaign minha-campanha --methods polygarbor polygarbor_aug polygarbor_p3 polygarbor_p3_aug polygarbor_p5 resnet18 yolo11n
cnn/.venv/bin/python comparison/suite.py --campaign minha-campanha --methods polygarbor polygarbor_aug polygarbor_p3 polygarbor_p3_aug polygarbor_p5 resnet18 yolo11n --scenarios source_a source_b --seeds 42
cnn/.venv/bin/python comparison/finish.py --campaign minha-campanha
```

`finish.py` também executa `localization.py`, que reutiliza os modelos completos das duas seeds para medir localização de tumor em mosaicos sintéticos 4×4. O threshold de Dice é escolhido na validação e aplicado uma única vez ao teste. Se quiser executar apenas essa etapa depois dos modelos completos, use `cnn/.venv/bin/python comparison/localization.py --campaign minha-campanha`.

O executor prepara caminhos locais das bibliotecas CUDA e inicia um processo por execução. As execuções são sequenciais para evitar competição entre modelos. Os pesos YOLO oficiais são baixados no primeiro uso. `prepare.py` reutiliza o TFDS já disponível em `polygarbor/data` e salva arrays em `comparison/cache`; não modifica o cache original. `dataset_manifest.json` contém classe, quantidade e SHA-256 de cada imagem.

`suite.py` aceita os métodos `polygarbor`, `polygarbor_aug`, `polygarbor_p3`, `polygarbor_p3_aug`, `polygarbor_p5`, `resnet18` e `yolo11n` por `--methods`, `--scenarios full few1 few2 few5 few10 few20 few50 few100 imbalance label_noise`, `--seeds 42 43` e `--epochs 100`. `--pilot --epochs 1 --scenarios few10 --seeds 42` serve para verificar a instalação; use outro nome de campanha para que esses resultados não entrem na comparação principal. Os runs existentes são preservados, inclusive falhas; para repetir uma falha, use uma nova campanha. Uma interrupção externa do executor pode deixar status `running`: consulte o log e não interprete isso como sucesso. Se o executor acompanha a saída não zero do filho, registra a falha. `finish.py` propaga falhas de testes e benchmarks e só reaproveita medições com sucesso registrado.

O treino completo e os sete tamanhos menores usam duas seeds. Desbalanceamento e ruído de rótulo usam a primeira seed. Cada treino é avaliado em treino, validação e teste. Os modelos treinados no conjunto completo recebem mais oito avaliações em teste com degradações fixas. Configuração, índices selecionados, rótulos alterados, hashes do código, versões de pacotes, histórico, tempos, recursos, predições individuais, matrizes e explicações ficam no diretório do run. Os modelos grandes e dados permanecem no disco e são ignorados pelo Git.

`finish.py` mede inferência em CPU com uma thread e afinidade restrita, CPU com quatro threads e GPU, em processos novos, sobre os modelos completos da seed 42. Depois executa testes e gera o relatório. `report.py` pode ser executado novamente a qualquer momento, pois só lê artefatos já salvos. Não execute testes ou benchmarks adicionais em paralelo com os treinos se for comparar os tempos.

```text
comparison/
  PROTOCOL.md
  dataset_manifest.json
  cache/                            # arrays compartilhados, ignorados pelo Git
  runs/2026-09-12/
    events.jsonl                    # comandos, códigos de saída, duração do processo
    full__resnet18__seed42/
      config.json / selection.json / status.json / run.log
      timings.json / resources.csv / resources.json
      model/ / model_size.json
      history.csv / history.json / epochs.jsonl
      eval/test_clean/metrics.json / predictions.npz / confusion_matrix.png
      explanations/
    edge/resnet18__cpu1/
  results/2026-09-12/
    REPORT.md / DISCUSSION.md
    run_index.csv / metrics.csv / learning_curve.csv / paired_tests.csv
    learning_curve.png / learning_curve_gmean.png / robustness.png / tradeoffs.png / edge.png
    stage_resources.csv / edge.csv / aggregation_comparison.csv
    error_gallery.png / source_shift_gallery.png / corruption_gallery.png
    localization/README.md / metrics.csv / localization_metrics.png / localization_examples.png
```

Interpretação: os cenários de poucos exemplos restringem o treino, mas mantêm 500 exemplos rotulados para validação. O ponto completo do PolyGabor preserva o limite padrão de 350 vetores/classe e registra quantos vetores entram no ajuste. ResNet começa do zero, YOLO usa pré-treino externo. Os mapas CAM, oclusão e similaridade são explicações computacionais, não segmentação. A simulação edge mede x86 com menos paralelismo; não demonstra desempenho em ARM/Jetson. Os códigos de origem recuperados dos nomes permitem testes agrupados adicionais, mas não identificam clinicamente pacientes. Para reproduzi-los, execute `cnn/.venv/bin/python comparison/prepare_groups.py` e depois `cnn/.venv/bin/python comparison/suite.py --campaign minha-campanha --scenarios source_a source_b --seeds 42`, antes de `finish.py`.

As variantes `p3` e `p5` usam 9 e 25 regiões por imagem; o banco de filtros permanece igual. `_aug` adiciona três vistas por original somente no treino. O teto de 350 vetores/classe permanece e o número de originais efetivamente retidos fica em `effective_training.json`. Consulte [G-mean versus macro-F1](GMEAN_VS_MACRO_F1.md) para interpretar as duas curvas sem confundir recall zero com falha de treinamento.

Campanha concluída: 138 execuções válidas, duas falhas conhecidas de ajuste PolyGabor com uma imagem por classe, 16 benchmarks de inferência e 16 execuções frias. As duas execuções interrompidas pela mudança da pasta foram arquivadas e repetidas com sucesso. A auditoria está em [integrity_audit.json](integrity_audit.json).
