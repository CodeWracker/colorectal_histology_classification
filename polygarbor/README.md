# polygarbor

Classificação de histologia colorretal combinando **banco de filtros de Gabor**
(textura) + **estatísticas no espaço Lab** (cor) como descritor, e um
**subespaço de Mahalanobis polinomial por classe** como classificador.

É a versão organizada, testada e executável do protótipo `garbor-downsampling.ipynb`.

---

## Instalação

O projeto usa [uv](https://docs.astral.sh/uv/) e vive inteiramente dentro desta pasta.

```bash
cd polygarbor

# inferência apenas (leve, sem TensorFlow)
uv sync

# + fluxo de dataset (baixa TensorFlow CPU e TFDS)
uv sync --extra dataset
```

Depois, todo comando roda com `uv run`:

```bash
uv run polygarbor --help
```

---

## Os dois fluxos

### Fluxo 1 — carregar o dataset (uma vez, permanente)

Baixa o `colorectal_histology` (5000 imagens 150×150, 8 classes) e o materializa
na pasta escolhida. O cache é permanente: execuções seguintes só leem do disco.

```bash
uv run polygarbor dataset \
  --data-dir ./data \
  --preview 9 \
  --export-per-class 1
```

A leitura é determinística por padrão (a ordem dos arquivos muda quais vetores
caem na amostragem de treino); use `--shuffle` se quiser variar a pré-visualização.

| opção | efeito |
| --- | --- |
| `--data-dir` | pasta permanente do TFDS (padrão `data`) |
| `--preview N` | salva uma grade com N amostras rotuladas |
| `--export-per-class N` | grava N PNGs por classe para testar o `predict` |
| `--export-split` | de qual split exportar (`train`/`val`/`test`) |
| `--shuffle` | embaralha a ordem de leitura (padrão: determinística) |

Em seguida, treine os subespaços:

```bash
uv run polygarbor train --data-dir ./data --model-dir ./artifacts/model --figures
```

Isso extrai os descritores de treino, ajusta um `PolyMahalanobis` por classe,
salva o modelo e avalia no split de validação (relatório, matriz de confusão em
CSV e PNG). Ajustes úteis: `--patches-per-row`, `--levels`, `--max-samples`,
`--limit` (roda com poucas imagens para um teste rápido) e `--skip-eval`.

Avaliação isolada de um modelo salvo:

```bash
uv run polygarbor evaluate --model-dir ./artifacts/model --split test
```

### Fluxo 2 — classificar uma imagem e gerar as visualizações

```bash
uv run polygarbor predict \
  -i artifacts/dataset/samples/00_tumor_1.png \
  --model-dir ./artifacts/model \
  --out-dir ./artifacts/predict \
  --true-label 0        # opcional: destaca acerto/erro
```

Saída no terminal: classe predita, confiança, ranking completo (similaridade,
votos e distância média por classe). Em `--out-dir/<nome-da-imagem>/`:

| arquivo | conteúdo |
| --- | --- |
| `prediction.json` | resultado completo, pronto para consumo por outro programa |
| `01_resultado.png` | imagem + ranking de classes |
| `02_banco_gabor.png` | os kernels do banco |
| `03_decomposicao_patch.png` | patch → cinza → mapas de energia de Gabor |
| `04_matriz_descritora.png` | heatmap da matriz (patches × features) |
| `05_mapas_similaridade.png` | mapas densos de similaridade por classe + vencedor por região |

`--no-viz` classifica sem gerar figuras; `--show` abre as janelas em vez de
apenas gravar os PNGs.

---

## Usar como biblioteca

O pacote é importável e **não exige TensorFlow** para classificar — o TFDS só é
carregado sob demanda pelo fluxo de dataset.

```python
from polygarbor import PolyGaborClassifier

clf = PolyGaborClassifier.load("artifacts/model")

pred = clf.predict("minha_lamina.png")        # caminho, ndarray ou tensor
print(pred.class_name, f"{pred.confidence:.1%}")
print(pred.ranking(clf.class_names))          # [(classe, similaridade, votos, distancia), ...]
print(pred.to_dict(clf.class_names))          # dict serializável em JSON
```

Treinando a partir dos seus próprios dados (qualquer iterável de
`(imagem, rótulo)`):

```python
clf = PolyGaborClassifier(class_names=["a", "b"], patches_per_row=3)
clf.fit([(img1, 0), (img2, 1), ...])
clf.save("meu_modelo")
```

Gerando as figuras programaticamente:

```python
from polygarbor import visualize

visualize.use_headless()                      # só grava PNG, não abre janela
pred = clf.predict(img, method="dense")       # o modo denso carrega a grade
fig = visualize.plot_similarity_maps(img, pred, clf.class_names)
visualize.save_figure(fig, "mapas.png")
```

Principais pontos da API:

| símbolo | papel |
| --- | --- |
| `PolyGaborClassifier` | pipeline completo: `fit`, `predict`, `predict_many`, `save`, `load` |
| `Prediction` | resultado: `label`, `class_name`, `confidence`, `votes`, `similarity`, `distances` |
| `GaborConfig` / `GaborBank` | parametrização e construção do banco de filtros |
| `patch_features` / `dense_features` | extração de descritores fora do classificador |
| `evaluate` / `EvaluationResult` | acurácia, relatório e matriz de confusão |

---

## Como funciona

1. **Filtragem global.** A imagem inteira é convertida para cinza e convoluída
   com cada par de kernels de Gabor (real e imaginário). A energia é a magnitude
   do par. Filtrar a imagem inteira antes de recortar evita artefato de borda
   nos patches.
2. **Descritor de 22 dimensões.** Por região: média e desvio de cada um dos 8
   mapas de energia (16 valores) + média e desvio dos 3 canais Lab (6 valores).
3. **Subespaço por classe.** Cada classe ganha um `PolyMahalanobis` ajustado
   sobre até `--max-samples` vetores, com expansão polinomial de `--levels` níveis.
4. **Decisão.** Cada unidade (patch ou pixel) vota na classe de menor distância;
   a imagem recebe a classe mais votada (`--aggregation voting`) ou a de menor
   distância média (`--aggregation mean`).

### `patch` vs `dense`

São duas escalas espaciais do **mesmo** descritor:

- `--method patch` (padrão) usa a grade de patches — a mesma escala do treino,
  e por isso é o modo correto para decidir a classe.
- `--method dense` calcula as estatísticas por pixel com janela deslizante numa
  grade 75×75. A escala local difere da vista no treino, então serve bem para
  **visualizar onde** a imagem se parece com cada classe, mas tende a ser menos
  fiel como decisão final. O `predict` sempre gera os mapas densos, mesmo quando
  a decisão vem do modo `patch`.

Com `--patches-per-row 1` (padrão, fiel ao protótipo) há um único patch por
imagem: o voto é trivial e a confiança sempre 100%. Use `--patches-per-row 3`
para que a votação entre patches passe a ter significado.

---

## Formato do modelo salvo

```
artifacts/model/
├── model.json            # configuração, classes e nomes das features
└── samples/class_NN.txt  # vetores de treino de cada classe
```

Os subespaços são **reconstruídos** a partir das amostras no `load()` — não há
pickle, então o modelo continua legível e portável entre versões da biblioteca.

---

## Resultados de referência

Configuração padrão (`--patches-per-row 1`, 8 filtros, 3 níveis, 350 amostras
por classe), treinada nos 4000 exemplos de treino:

| split | acurácia |
| --- | --- |
| validação (500 imagens) | 0.828 |
| teste (500 imagens) | 0.786 |

`adipose` (recall 1.00), `lympho` (0.94) e `tumor` (0.87) são as classes mais
sólidas. As duas fracas são `mucosa` (recall 0.45, confundida com `debris`) e
`complex` (0.51, confundida com `stroma`) — ambas com precisão alta, ou seja, o
modelo é conservador ao atribuí-las.

Como o treino é determinístico (sem embaralhamento e com `--seed` fixo), repetir
o comando reproduz exatamente o mesmo modelo.

---

## Testes

```bash
uv run pytest
```

A suíte usa imagens sintéticas e cobre descritor, treino, ida-e-volta do
`save`/`load`, predição nos dois modos e geração de todas as figuras — sem
precisar do dataset nem do TensorFlow.

---

## Estrutura

```
src/polygarbor/
├── gabor.py       # GaborConfig e GaborBank
├── features.py    # descritor por patch e denso
├── classifier.py  # PolyGaborClassifier e Prediction
├── dataset.py     # carga permanente via TFDS (import preguiçoso)
├── evaluation.py  # métricas por imagem
├── visualize.py   # todas as figuras
├── console.py     # saída formatada no terminal
└── cli.py         # argparse: dataset / train / evaluate / predict / info
```
