# Como interpretar G-mean multiclasse e macro-F1

As duas métricas dão importância a todas as classes, mas respondem a perguntas diferentes. Macro-F1 equilibra precisão e recall de cada classe; G-mean multiclasse mede o equilíbrio dos recalls e penaliza fortemente a classe menos reconhecida.

## Definições

Para cada classe $i$, $Recall_i = TP_i/(TP_i+FN_i)$ e $Precision_i = TP_i/(TP_i+FP_i)$. Nesta comparação, $C=8$.

$$G\text{-mean} = \left(\prod_{i=1}^{C} Recall_i\right)^{1/C}$$

$$Macro\text{-}F1 = \frac{1}{C}\sum_{i=1}^{C}\frac{2\,Precision_i\,Recall_i}{Precision_i+Recall_i}$$

O cálculo do G-mean usa `exp(mean(log(recalls)))` para estabilidade numérica quando todos os recalls são positivos. Se qualquer recall for zero, retorna exatamente zero. Não adicionamos epsilon nem correção que esconda uma classe completamente ignorada. Quando uma classe não tem exemplos reais no conjunto avaliado, seu recall não é estimável e o G-mean das oito classes é registrado como `null`, em vez de inventar um valor. Isso ocorre nas validações dos splits por origem, que não contêm todas as classes; seus testes contêm as oito.

## O que cada métrica revela

| Propriedade | Macro-F1 | G-mean multiclasse |
| --- | --- | --- |
| Usa precisão | Sim, explicitamente | Não explicitamente; falsos positivos prejudicam o recall das classes de origem desses erros |
| Usa recall | Sim | Sim, exclusivamente |
| Agregação | Média aritmética dos F1 por classe | Média geométrica dos recalls |
| Uma classe nunca é reconhecida | Contribui com F1 zero, mas o resultado agregado pode continuar razoável | Todo o resultado se torna zero |
| Troca entre classes fortes e fracas | Ganhos nas fortes podem compensar parte da perda nas fracas | Penaliza mais desequilíbrios e recalls próximos de zero |
| Pergunta prática | O modelo mantém precisão e cobertura equilibradas por classe? | O modelo consegue reconhecer todas as classes sem deixar alguma para trás? |

Por exemplo, considere recalls de sete classes iguais a 0,95 e recall da oitava igual a zero. O G-mean é zero, mesmo que a acurácia seja alta. Já o macro-F1 pode ser relativamente alto, dependendo das precisões; não é possível calculá-lo somente a partir dos recalls. Se o recall da oitava classe aumentar para 0,10, o G-mean sobe para aproximadamente 0,717. Isso mostra como a passagem de nenhum acerto para poucos acertos de uma classe pode produzir um salto na curva. Esses números são um exemplo matemático, não resultados de um treinamento.

## Como ler as curvas de poucos exemplos

O eixo X conta imagens originais distintas usadas no treino, com indicação por classe e do total. Augmentations e patches não aumentam esse orçamento: uma imagem dividida em nove regiões continua contando como uma imagem original. O eixo Y apresenta macro-F1 ou G-mean no mesmo conjunto de teste. As curvas usam os mesmos subconjuntos aninhados e seeds; a faixa representa mínimo e máximo entre as seeds, não um intervalo de confiança estatístico.

Um macro-F1 crescente acompanhado de G-mean zero indica que o modelo melhorou em algumas classes, mas ainda não reconhece pelo menos uma. G-mean crescente com macro-F1 pouco alterado pode indicar melhora da cobertura da classe mais fraca, compensada por falsos positivos. Valores bons das duas métricas sugerem um equilíbrio melhor, que deve ser confirmado pelos recalls e precisões individuais e pela matriz de confusão.

Com poucos exemplos, G-mean pode variar bastante entre seeds, especialmente perto de recall zero. Uma curva em zero não significa que todas as predições estão erradas. Uma falha de ajuste do classificador também não é G-mean zero: nesse caso não há modelo avaliado e o ponto fica ausente, com a falha registrada no índice.

## Cuidados nesta comparação

Os cenários de escassez mantêm 500 exemplos rotulados de validação: restringem o treino, não todo o orçamento de anotação. A comparação entre splits aleatórios e splits por origem tem populações e quantidades diferentes. A classe `empty` está concentrada em duas origens, o que afeta tanto a interpretação de generalização quanto a possibilidade de cobrir todas as classes em três splits sem compartilhar origens.

G-mean não mede calibração, qualidade da explicação, tempo, memória ou risco clínico. Tampouco prova generalização entre pacientes. O relatório apresenta essas dimensões separadamente. A implementação está em `cnn/src/cnn/evaluation.py`, e os valores anteriores foram recalculados a partir das predições salvas, sem retreinar modelos.

## Exemplos medidos nesta campanha

Na seed42, PolyGabor com cinco imagens originais por classe obteve macro-F1 de aproximadamente 0,431, mas G-mean zero. Isso significa que houve aprendizado útil de algumas classes, porém pelo menos uma não recebeu nenhum acerto. No teste de origem source_b, a ResNet alcançou macro-F1 de aproximadamente 0,344 e também G-mean zero. Os resultados e os recalls exatos por classe estão nos respectivos `eval/test_clean/metrics.json`.

Esses exemplos justificam apresentar as duas curvas: macro-F1 permite acompanhar ganhos graduais de precisão e recall, enquanto G-mean torna explícita a falta de cobertura de uma classe. Os valores são de execuções específicas, não médias sobre todas as configurações.
