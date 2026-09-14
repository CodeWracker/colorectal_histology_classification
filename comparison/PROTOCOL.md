# Protocolo registrado antes dos experimentos

Comparar PolyGabor padrão (1 patch, 3 níveis, máximo 350 vetores/classe), ResNet-18 do notebook (do zero) e YOLO11n-cls (transferência ImageNet). Preservar todos os dados e artefatos existentes.

- TFDS colorectal_histology 2.0.0, ordem determinística, splits originais 4000/500/500. Mesmos exemplos e rótulos para todos os métodos.
- Salvar índices, hashes SHA-256 das imagens e contagens por classe. Verificar duplicatas exatas entre splits. O TFDS não fornece paciente/lâmina: teste interno em recortes, sem alegar generalização entre pacientes.
- Treino completo, 1, 2, 5, 10, 20, 50 e 100 exemplos/classe (subconjuntos aninhados), desbalanceamento (10% das classes tumor/complex/stroma/mucosa), ruído de rótulo uniforme em 20% do treino; seeds 42 e 43 nos cenários de escassez para medir sensibilidade à amostragem/inicialização.
- ResNet: 128 px, batch 32, Adam 1e-3, até 100 épocas, early stopping val_accuracy/paciência 8 e redução LR val_loss/paciência 4. Flips horizontal/vertical e brilho ±0.1 como no notebook.
- YOLO: 128 px, batch 32, até 100 épocas e paciência 8; preservar transformações próprias do backend e documentar diferenças.
- Seleção exclusivamente pela validação. Sem ajustar hiperparâmetros no teste.
- Teste limpo completo e perturbações fixas: desfoque, ruído, brilho, alteração de cor, compressão JPEG, baixa resolução, oclusão e rotação. Mesmos pixels perturbados para todos. Não equivalem a outra base clínica.
- Acurácia, macro-F1, balanced accuracy, MCC, kappa, precisão/recall/F1 por classe, matriz de confusão, top-2, AUROC OVR, NLL, Brier e ECE. Similaridade normalizada PolyGabor é um escore heurístico, não probabilidade calibrada; confiança de votação não entra como probabilidade.
- Bootstrap pareado no teste para diferença de acurácia e McNemar exato; IC de recortes não mede incerteza entre pacientes; testes exploratórios.
- Processos separados por run. Etapas: leitura, preparação/extração, construção, ajuste, salvamento, recarga, avaliação e explicações. CPU em core-% (100% = um núcleo), RAM RSS amostrada, CPU-segundos, GPU-% global e VRAM por PID. GPU global inclui desktop; memória medida não é consumo exclusivo do modelo.
- Inferência aquecida batch 1 (latência mediana/p95), throughput em lote, carga fria em subprocesso, CPU e GPU para CNN. MB = 1.000.000 bytes; reportar pacote de inferência separado do checkpoint com otimizador.
- Figuras: curvas, matrizes, robustez, trade-offs, casos concordantes e discordantes, CAM da ResNet e mapas/descritores do PolyGabor. Mapas não são segmentação nem comprovação de validade clínica.

Cada execução terá config, status, log, tempos, recursos, modelo, histórico, predições individuais e figuras. Falhas e pilotos serão registrados e excluídos das médias principais. Resultados serão consolidados em CSV e Markdown.

Extensão solicitada antes dos experimentos principais: curva macro-F1 versus exemplos por classe (subconjuntos aninhados, seeds 42/43). Edge será uma simulação em CPU x86 com afinidade restrita a um processador lógico e uma thread, batch 1, comparada com CPU com quatro threads e GPU. Não representa medições em Raspberry Pi, ARM ou Jetson. Memória não será artificialmente limitada; os picos indicarão se caberiam em orçamentos hipotéticos. Treinamento em edge será discutido a partir do custo medido, sem extrapolar tempo de GPU para ARM.

## Extensão de generalização por origem

Uma inspeção posterior dos arquivos originais revelou dez códigos CRC-Prim-HE nos nomes dos recortes. Eles serão recuperados por correspondência exata de SHA-256 dos pixels, sem alterar os splits da campanha principal. Dois novos splits foram fixados antes de treinar os modelos agrupados: source_a treina 01–07, valida em 08 e testa 09/10; source_b treina 01–05/07/10, valida em 08 e testa 06/09. Cada um terá os três métodos e seed42, sem ajuste de hiperparâmetros usando seus testes. Os resultados não serão misturados às médias dos splits aleatórios.

A classe empty ocorre somente nas origens 06 e 10, impossibilitando cobertura das oito classes simultaneamente em treino, validação e teste sem compartilhar origens. Os splits escolhidos preservam todas as classes no treino e teste; a validação 08 não contém adipose/empty. Essa restrição e a diferença de tamanho e composição dos testes serão relatadas. Os códigos indicam origem/lâmina inferida pelo nome, não identidade clínica confirmada de paciente. O artigo descreve dez lâminas independentes: https://pmc.ncbi.nlm.nih.gov/articles/PMC4910082/.

## Extensão leave-one-source-out

Uma revisão de `source_a` e `source_b` encontrou três limitações. Primeiro, eram duas partições escolhidas manualmente, com uma seed cada. Segundo, `source_b` concentra 590 dos 1.403 recortes de teste na classe empty, com apenas 35 exemplos dessa classe no treino, todos da origem 10. Terceiro, a validação na origem 08 não contém adipose nem empty, enquanto as CNNs usam early stopping por val_accuracy. Esses cenários passam a ser registro histórico e não sustentam comparação entre métodos.

Antes de qualquer treino agrupado novo, `loso.py` fixou dez folds leave-one-source-out em `loso_manifest.json`. Cada fold testa uma origem inteira. A validação é 10% de cada classe das nove origens restantes, com mínimo de um recorte e seed de partição 20260913. O treino usa o restante. Os índices são os mesmos para todos os métodos e seeds. Serão executados polygarbor, resnet18, resnet18_imagenet, yolo11n_random e yolo11n, com os hiperparâmetros, critérios de parada e teto de 350 vetores/classe da campanha principal, nas seeds 42 e 43. Nenhum hiperparâmetro será ajustado pelos testes.

A métrica primária é o macro-F1 sobre as 5.000 predições concatenadas dos dez folds, em que cada recorte é predito por um modelo que não viu sua origem. Também serão reportados G-mean, balanced accuracy, recall por classe e macro-F1 sem os recortes empty. Serão reportados ainda o macro-F1 por fold somente nas classes presentes e a diferença pareada entre PolyGabor e cada CNN, com bootstrap sobre as dez origens. A validação vem das mesmas origens do treino, mas o teste permanece disjunto por origem. Os dados vêm de dez imagens de um único instituto e scanner, portanto o resultado mede generalização entre imagens desse acervo, não validação externa.

Para caber no disco, a cópia de imagens `yolo_dataset` de cada run YOLO LOSO é removida após a conclusão. Ela é derivada integralmente do cache e dos índices do manifesto.

## Extensão de quantização pós-treino das CNNs

Antes da execução, foi fixada uma comparação com versões quantizadas somente para inferência das CNNs, sem retreino e sem quantização durante o treino. O PolyGabor permanece em precisão cheia como referência. Entram os modelos completos de resnet18, resnet18_imagenet, yolo11n_random e yolo11n nas seeds 42 e 43.

Todas as variantes usam LiteRT (TFLite) e o mesmo interpretador ai-edge-litert com XNNPACK. A ResNet Keras é exportada como SavedModel com lote fixo 1 e convertida pelo conversor do TensorFlow 2.20; a YOLO é convertida do PyTorch por litert-torch. A partir de cada modelo float32, ai-edge-quantizer gera três variantes: pesos float16 (float casting), int8 dinâmico (pesos int8 por canal e ativações quantizadas em tempo de execução) e int8 estático (pesos e ativações int8, com entrada e saída float32). A calibração usa 200 recortes do split de treino, 25 por classe com seed 20260914; validação e teste não são usados.

O pré-processamento de inferência não depende de TensorFlow nem de PyTorch: redimensionamento bilinear do OpenCV para a ResNet, com a normalização ImageNet quando aplicável, e redimensionamento bilinear do Pillow para a YOLO, equivalente à transformação salva no checkpoint. A variante float32 separa o efeito de runtime e pré-processamento do efeito da quantização. Sua concordância com as predições originais no teste limpo deve ser de pelo menos 99%; abaixo disso a conversão é considerada inválida e a execução é interrompida.

As variantes serão avaliadas no teste limpo e nas oito perturbações fixas, com os mesmos pixels da campanha. Serão reportados tamanho do arquivo, acurácia, macro-F1, G-mean, balanced accuracy, recall por classe, concordância com o modelo original e com a variante float32, e diferença de macro-F1. Latência mediana e p95 de batch 1, incluindo pré-processamento, e pico de RSS serão medidos em processos novos nos modos cpu1 e cpu4 do benchmark edge, para a seed 42, sem treinos em execução.

O ambiente fica em `comparison/quantization`, com lock próprio, porque litert-torch exige torch 2.11 e alteraria versões travadas no ambiente da campanha. Os modelos treinados com torch 2.7.1 são carregados sem alteração.

## Extensão G-mean, augmentation e grade de patches

A pedido do usuário, a comparação acrescenta G-mean multiclasse exato, sem suavização de recall zero, recalculado das predições existentes. Valores sem suporte em todas as classes serão indefinidos. O gráfico contará imagens originais, e um documento separado explicará a diferença para macro-F1.

Quatro variantes adicionais PolyGabor: grade 3×3 (9 descritores), grade 5×5 (25 descritores), grade 1×1 com augmentation e grade 3×3 com augmentation. O banco mantém oito pares de kernels: quantidade de regiões não é quantidade de filtros. Augmentation é aplicada somente no treino: original mais três variantes determinísticas com flips horizontal/vertical e brilho ±0,1, recortado ao intervalo RGB válido. É uma política controlada próxima à ResNet, não uma cópia do RandAugment da YOLO. As variantes preservam níveis=3, teto=350 vetores/classe e votação, registrando tanto vetores extraídos quanto número de imagens originais efetivamente retidas. As curvas e cenários usam os mesmos índices/seeds das outras abordagens.

As quatro variantes serão executadas no treino completo e nos sete tamanhos de escassez com duas seeds; desbalanceamento, ruído de rótulo e dois splits de origem usarão seed42. Isso permite separar o efeito da grade do efeito de augmentation sem inflar artificialmente o eixo de quantidade de dados.

## Localização quantitativa em mosaicos

Antes da execução, foi fixado um teste sintético de localização binária tumor versus demais classes. Serão construídos 20 mosaicos de validação e 30 de teste, cada um com grade 4×4 de patches de 150×150 pixels, resultando em 600×600 pixels. Cada mosaico terá dois patches de tumor em posições aleatórias e quatorze patches das demais classes. Não haverá reutilização de patches dentro de cada split; validação e teste continuam disjuntos conforme os splits originais. O layout, índices e rótulos serão salvos.

Serão reutilizados os modelos completos das seeds 42 e 43, sem retreino nem seleção pelo teste. Para PolyGabor, o escore espacial será `-log1p(distância para tumor)` calculado pelo descritor denso 75×75. Para ResNet-18, será usado o CAM exato da cabeça global-average-pooling/linear, antes do softmax e após ReLU. A rede será estendida de forma totalmente convolucional para entrada 600×600 com os mesmos pesos; equivalência numérica em entrada 128×128 será verificada. Ambos os mapas serão interpolados linearmente para 600×600.

O threshold que maximiza Dice agrupado será escolhido exclusivamente nos mosaicos de validação de cada método e seed. No teste serão reportados AUROC e average precision em pixels, Dice e IoU no threshold fixado, além de médias por mosaico com intervalo bootstrap de 95% reamostrando mosaicos inteiros. Pixels vizinhos não serão tratados como réplicas independentes para o intervalo. A prevalência positiva é 12,5%; o baseline de marcar tudo como tumor tem Dice 0,2222 e AUROC 0,5.

A comparação mede localização em mosaicos artificiais feitos de patches homogêneos rotulados. As bordas da máscara vêm da montagem, não de anotação histopatológica pixel a pixel. O resultado não demonstra segmentação de tumor em lâminas reais. A parte qualitativa nas dez imagens grandes depende da disponibilidade local do dataset `colorectal_histology_large` e será separada do teste quantitativo.

### Revisão metodológica após inspeção visual

A primeira execução revelou que calcular o mapa no mosaico completo permitia que o campo receptivo e a interpolação cruzassem bordas artificiais entre imagens independentes. Esse desenho foi substituído pela versão 2 do protocolo. Cada patch de 150×150 agora é explicado isoladamente na resolução de entrada normal do método; o mapa é redimensionado somente dentro do próprio patch e os dezesseis mapas são então remontados sem mistura nas bordas.

A unidade primária de avaliação passa a ser o patch, pois esse é o nível da verdade fornecida pelo dataset. O escore explicativo de um patch é a média do seu mapa. Serão reportados AUROC e AP entre patches, recuperação dos dois tumores entre os dois maiores escores, taxa de mosaicos com ambos recuperados e Dice/IoU por patch com threshold escolhido apenas na validação. O escore de classificação de tumor em cada patch também será avaliado separadamente para distinguir erro do classificador de erro da explicação.

As métricas em pixels passam a ser chamadas de localização de região fracamente anotada e ficam como análise secundária. Todo pixel de um patch tumor recebe rótulo positivo apenas por herança da classe da imagem; portanto essas métricas não comprovam localização interna nem segmentação histológica. As figuras marcarão os patches tumorais em verde, os dois patches mais destacados pelo mapa em amarelo e o contorno do threshold em ciano, calculado separadamente dentro de cada patch.

## Controle de inicialização das CNNs

Antes desta extensão foram acrescentados dois controles para separar arquitetura e pretraining: `resnet18_imagenet` e `yolo11n_random`. Eles executarão os mesmos splits, subconjuntos aninhados, seeds, perturbações, métricas, monitoramento e critérios de parada de suas arquiteturas correspondentes. Serão incluídos treino completo nas seeds 42/43, sete cenários de escassez nas duas seeds, desbalanceamento e ruído de rótulo na seed 42 e os dois splits por origem na seed 42.

`resnet18_imagenet` recebe os 20 kernels convolucionais e 20 conjuntos de BatchNorm de `Torchvision ResNet18_Weights.IMAGENET1K_V1`; a cabeça de oito classes é aleatória, todas as camadas são ajustáveis e a entrada usa média/desvio ImageNet. O port mantém a arquitetura Keras usada pelo projeto e registra a origem em `initialization.json`. Como padding e parâmetros numéricos do BatchNorm não são uma reprodução byte a byte do runtime PyTorch, o experimento mede inicialização com filtros ImageNet portados, não equivalência exata de logits com Torchvision.

`yolo11n_random` constrói `yolo11n-cls.yaml` e chama o treino com `pretrained=False`. `yolo11n` continua partindo de `yolo11n-cls.pt`; arquitetura, tamanho de entrada, política de augmentation, otimizador automático e demais argumentos permanecem iguais. Na YOLO, a origem dos pesos é o fator controlado pretendido. Na ResNet, a configuração ImageNet também exige média/desvio ImageNet, enquanto a variante original usa escala 0–1; portanto seu contraste combina inicialização e normalização compatível e não identifica causalmente somente os pesos. Comparações entre ResNet e YOLO continuam incluindo diferenças de framework e política de treino.

A localização versão 3 acrescenta o CAM de `resnet18_imagenet` nos mesmos mosaicos, seeds e métricas da ResNet aleatória. Os mapas PolyGabor e ResNet aleatória da versão 2 serão reutilizados sem recomputação; seus artefatos brutos anteriores serão preservados com sufixo `_v2`. Uma figura dedicada comparará, para cada inicialização ResNet, os escores classificatórios por patch e o CAM, evitando confundir a decisão correta com uma explicação espacial correta.
