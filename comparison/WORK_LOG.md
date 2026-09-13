# Registro de execução

## Preparação

Inspecionados `cnn_test.ipynb`, a CLI e API PolyGabor, seus testes e os artefatos existentes. O notebook contém ResNet-18 treinada do zero e YOLO11n-cls com pesos externos. O cache TFDS local já tinha as 5000 imagens. Hardware identificado: Intel Core Ultra 9 185H, 22 processadores lógicos, aproximadamente 32 GB RAM e NVIDIA RTX 4070 Laptop GPU de 8 GB compartilhada com o desktop.

O sandbox não conseguia iniciar comandos (`bwrap: loopback: Failed RTM_NEWADDR`). As leituras, instalações e execuções autorizadas foram feitas pela alternativa de execução com revisão automática. Isso também afetou o visualizador de imagens; as figuras foram inspecionadas por leitura e codificação da prévia, preservando os arquivos originais.

O protocolo foi escrito antes dos experimentos principais e ampliado a pedido do usuário com curva 1/2/5/10/20/50/100 exemplos por classe e simulação edge. Markdown passou a ser escrito com cada parágrafo em uma única linha. Os splits foram materializados em arrays compartilhados em 5,33 s; auditoria SHA-256 não encontrou duplicatas exatas. O manifesto registra as 5000 imagens em ordem.

## Ambiente e pilotos

Criado ambiente isolado em `cnn/.venv`. A resolução inicial trouxe bibliotecas CUDA incompatíveis entre si; foram fixados TensorFlow 2.20.0, PyTorch 2.7.1 e torchvision 0.22.1, com lockfile uv. As bibliotecas CUDA 12 foram reinstaladas após remover pacotes CUDA 13 com arquivos sobrepostos. Links locais e `LD_LIBRARY_PATH` da campanha resolveram a descoberta das bibliotecas, sem alteração do driver.

`pilot-v1`: PolyGabor few10/seed42 concluiu em 11,6 s. ResNet falhou porque TensorFlow ainda não descobria a GPU. A falha está preservada com traceback e status; não entrou na comparação principal.

`pilot-v2`: ResNet few10/seed42/uma época concluiu em 34,8 s e YOLO em 23,8 s, incluindo toda a pipeline do piloto. Ambos usaram a GPU. Esses números servem para verificação de funcionamento, não para comparar convergência. A primeira época da ResNet inclui a compilação inicial.

Testes pequenos iniciais passaram (4 testes). A primeira tentativa de coletar também os testes PolyGabor encontrou colisão entre arquivos `test_pipeline.py`; a execução final usa `--import-mode=importlib`. Essa tentativa falhou na coleta em 1,15 s, no início da campanha, antes do ajuste principal. Não foram mantidos testes ou outros treinos em paralelo com as medições.

## Revisão independente

A revisão apontou que Ultralytics ignora `batch` para uma lista NumPy inteira; a inferência passou a dividir explicitamente a lista em lotes. Foram corrigidos também a omissão da seed 43 no treino completo e os parâmetros ResNet aceitos sem efeito pela YOLO, que agora são rejeitados quando não se aplicam. O limite padrão de 350 vetores/classe PolyGabor foi preservado e passou a ter registro explícito do orçamento efetivo.

Uma segunda revisão identificou redefinição interna de threads pelo Ultralytics e acesso tardio a imagens via memmap no benchmark edge. O limite de threads é reaplicado após inicialização do backend; as imagens edge são copiadas para RAM antes da medição. Os experimentos de treino e edge continuam sequenciais.

## Resultados parciais

Primeiro PolyGabor completo, seed42: acurácia de teste 0,786 e macro-F1 0,7802, reproduzindo a referência existente. Extração de descritores: 20,72 s; ajuste: 0,47 s; pacote completo de avaliações e explicações: 249,5 s. O custo total inclui avaliação de treino, validação, nove condições de teste e figuras. Matriz e mapas já foram inspecionados: `complex` e `mucosa` têm os menores recalls, enquanto as respostas Gabor destacam frequências e orientações locais.

Primeira ResNet completa, seed42: early stopping após 26 épocas, restaurando o melhor estado de validação; acurácia de teste 0,906 e macro-F1 0,9052. A validação oscilou bastante no começo do treino. O desempenho caiu particularmente na alteração de cor, mostrando que a vantagem no teste limpo não se conserva automaticamente em todas as degradações. Essa observação não levou a ajuste de hiperparâmetros.

O andamento posterior é registrado em `campaign.log` e `runs/2026-09-12/events.jsonl`; `epochs.jsonl`, `history.csv` e `timings.json` guardam etapas parciais de cada run. As tabelas finais são reconstruídas diretamente dos arquivos de métricas.

## Recuperação após mudança de diretório e extensão das métricas

O usuário moveu o repositório para `.../ufsc/viscomp/colorectal_histology_classification` e depois de volta. Isso interrompeu duas execuções YOLO: label_noise e source_b. Seus logs mostram FileNotFoundError ao salvar artefatos nos caminhos absolutos antigos. As tentativas foram marcadas como interrompidas e preservadas em `runs/interrupted`, e somente essas duas serão repetidas. A auditoria verificou 56 runs completos, incluindo correspondência dos rótulos e ordem de teste, coerência das métricas, presença de recursos e tamanho dos modelos; todos passaram.

As duas falhas few1 do PolyGabor original são separadas desse incidente: a biblioteca lança IndexError ao ajustar uma única linha por classe. Elas permanecem como falhas observadas da implementação, sem substituição por F1/G-mean zero. As variantes com patches e augmentation testarão se múltiplos descritores derivados de uma imagem mudam esse comportamento.

G-mean foi acrescentado e recalculado das predições salvas, sem retreinar. A documentação `GMEAN_VS_MACRO_F1.md` distingue média geométrica de recalls e média aritmética de F1. O primeiro gráfico revelou que ResNet pode manter G-mean zero com poucos exemplos, mesmo quando o macro-F1 já sobe. Todos os 19 testes passaram após corrigir a importação no modo importlib.

Foram enfileiradas quatro ablações PolyGabor, com grades de 9/25 regiões e augmentation determinística. Revisão independente confirmou alinhamento dos rótulos aos patches, reprodutibilidade da seleção, preservação da decisão por votos nas métricas e tratamento de recall zero/ausente. O relatório também passou a exibir quantas imagens originais distintas efetivamente entram no ajuste após o teto de vetores.

## Conclusão da campanha e revisão dos artefatos

As duas execuções YOLO interrompidas foram repetidas e concluíram com sucesso. A extensão também concluiu as 80 execuções das quatro variantes PolyGabor. Resultado final: 138 execuções completas e auditadas, duas falhas conhecidas few1 do PolyGabor original, 16 benchmarks edge e 16 subprocessos frios com retorno zero. Nenhum treinamento permanece pendente. A soma dos tempos de processo nos 140 eventos finais é aproximadamente 3,50 horas; não inclui toda a preparação, pilotos, interrupções ou benchmarks edge.

O modelo ResNet antigo continha momentos do Adam porque clone_model copiou o estado de compilação. Os modelos foram exportados como Functional não compilados, mantendo os checkpoints originais fora do diretório de inferência. A exportação do treino completo seed42 reduziu o artefato para 45,009 MB e produziu diferença máxima de probabilidade zero; cada conversão guarda inference_export.json. As medições edge usam os modelos de inferência corrigidos.

Foram inspecionados matrizes, curvas, mapas Gabor/CAM/oclusão e galerias de erros, perturbações e origens. Legendas sobrepostas nas galerias foram corrigidas; mapas de oclusão agora mostram sinal e magnitude, evitando interpretar uma queda de escore quase nula como evidência forte. A discussão final diferencia observação, hipótese e limitação experimental.

A revisão final identificou que finish.py podia ignorar saída não zero de testes/benchmarks. O executor agora propaga falhas, permite completar medição fria faltante sem repetir a quente válida, e o relatório exclui tempos de subprocessos fracassados. A regressão foi verificada: 20 testes passaram. Também passaram oito comandos reais de CLI, incluindo info/predict/evaluate de ambas as CNNs, ajuda de dataset e info do PolyGabor. A primeira tentativa do smoke usou equivocadamente o interpretador base ao resolver o symlink do venv; foi corrigida para o caminho absoluto do venv, sem alterar os modelos. Logs finais estão em results/2026-09-12/cli_smoke.

Resultados interpretados em results/2026-09-12/DISCUSSION.md; tabelas reproduzíveis em REPORT.md; G-mean explicado separadamente em GMEAN_VS_MACRO_F1.md. stage_resources.csv consolida as medições por etapa. Os arquivos Markdown mantêm cada parágrafo em uma linha, conforme solicitado.

## Localização quantitativa

Foi fixado antes da execução um protocolo com 20 mosaicos de validação e 30 de teste, dois patches de tumor por mosaico 4×4 e ausência de repetição dentro de cada split. Os modelos completos das seeds 42/43 foram reutilizados. Thresholds de Dice foram escolhidos somente na validação; AUROC, AP, Dice e IoU foram calculados em pixels, com intervalos por bootstrap de mosaicos.

Depois da inspeção da primeira figura, a versão 1 foi substituída porque mapas calculados no mosaico inteiro cruzavam bordas artificiais. A versão 2 explica cada patch isoladamente e remonta os mapas sem mistura, marca a verdade, o top-2 e o threshold na figura e usa o patch como unidade primária. A versão original permanece descrita no protocolo para registrar a revisão.

PolyGabor obteve AUROC de explicação por patch 0,8274/0,8233 e recall dos tumores no top-2 46,7%/45,0%; a ResNet obteve 0,9703/0,9860 e 78,3%/85,0%. O AUROC do classificador por patch foi 0,9638/0,9641 no PolyGabor e 0,9981/0,9984 na ResNet, mostrando que a conversão da decisão em explicação espacial perde mais informação no PolyGabor. Os mapas, layouts, escores de classificação, thresholds, recursos e métricas foram salvos em runs/2026-09-12/localization e results/2026-09-12/localization.

A inspeção visual da versão 2 confirmou que os contornos não atravessam as bordas computacionalmente. Nos três exemplos fixos, a ResNet recuperou ambos os tumores e o PolyGabor recuperou zero, um e zero. O teste mede identificação de patches e concentração fraca de evidência, não segmentação de tecido contínuo. O dataset local não contém `colorectal_histology_large`, portanto a extensão qualitativa nas dez imagens grandes permanece explicitamente separada e não executada.
