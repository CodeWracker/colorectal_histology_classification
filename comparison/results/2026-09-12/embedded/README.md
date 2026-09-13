# Pré-avaliação de inferência em microcontroladores

Esta análise compara o modelo `full__polygarbor__seed42` com os limites publicados para três placas ou microcontroladores. Ela mede armazenamento e arrays reais do modelo carregado, estima uma representação quantizada e conta operações do banco de Gabor. Ela não executa o firmware do alvo, não prevê latência e não mede energia.

| Alvo | Flash | SRAM | artefato salvo/flash | estado ajustado/flash | int8/flash | mapas atuais/SRAM | fluxo mínimo/SRAM | avaliação |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ESP32-WROOM-32 | 4,194,304 B | 532,480 B | 15.5% | 85.5% | 21.7% | 220.8% | 5.1% | plausível somente após porte C/C++, quantização e extração em fluxo |
| Arduino Uno R3 | 32,768 B | 2,048 B | 1986.1% | 10947.7% | 2775.9% | 57408.4% | 1325.7% | inviável para o método atual, mesmo na estimativa quantizada |
| PIC16F877A | 14,336 B | 368 B | 4539.8% | 25023.3% | 6344.9% | 319490.2% | 7377.7% | inviável para o método atual, mesmo na estimativa quantizada |

O diretório salvo ocupa 0.651 MB, mas contém 2.800 vetores de treino em texto. Ao carregar, a biblioteca reconstrói bases polinomiais e mantém as amostras: os arrays ocupam 3.857 MB. As amostras não são consultadas por evaluate() e podem ser excluídas de um exportador. Com kernels, uma exportação de inferência float32 ocuparia pelo menos 3.587 MB, antes do firmware. A estimativa int8/uint16 cai para 0.910 MB, mas sua equivalência preditiva ainda precisa ser testada.

A extração atual mantém RGB, cinza float32, Lab float32 e oito mapas de energia float32, um piso de 1.176 MB sem contar temporários do OpenCV. Uma implementação em fluxo usaria aproximadamente 3.6 kB para extração. Somado a um espaço de trabalho float32 conservador de 23.5 kB para avaliar uma classe por vez, o total estimado é 27.1 kB. Esse núcleo precisaria ser escrito em C/C++ com aritmética fixa ou cuidadosamente quantizada.

O banco executa duas convoluções 21×21 para cada um dos oito filtros. Uma imagem 150×150 exige aproximadamente 158,760,000 multiplicações-acumulações no cálculo direto, além de magnitude, estatísticas Lab e distância polinomial. Por isso, razão de frequências de clock não é uma estimativa válida de latência.

O ESP32 é o único dos três alvos com uma rota plausível para a inferência completa: exportar o estado ajustado sem reconstrução, quantizar, manter coeficientes na flash e processar a imagem em fluxo. O Arduino Uno e o PIC16F877A não comportam sequer a estimativa int8 do estado polinomial na memória de programa; para eles seria necessário trocar o classificador por uma aproximação muito menor, reduzir filtros e resolução ou usá-los apenas como controladores de um coprocessador.

Treinar no dispositivo não é uma meta realista nos três alvos. A etapa usa amostras por classe, expansões polinomiais e decomposições numéricas, com memória de trabalho muito acima da inferência. A avaliação embarcada defensável é treinar externamente, exportar uma representação de inferência e medir fidelidade, latência, RAM, flash e energia no alvo.

As especificações usadas são as folhas oficiais do [ESP32-WROOM-32](https://documentation.espressif.com/esp32-wroom-32_datasheet_en.html), do [Arduino UNO R3](https://docs.arduino.cc/resources/datasheets/A000066-datasheet.pdf) e do [PIC16F87XA](https://ww1.microchip.com/downloads/en/devicedoc/39582b.pdf).

## Próxima etapa necessária

A simulação fiel exige um exportador de coeficientes e um núcleo de inferência C/C++. Depois disso, a sequência correta é verificar concordância de classes e degradação de macro-F1/G-mean no teste inteiro, compilar com o toolchain de cada alvo, obter mapa de memória e contagem de ciclos no simulador e medir latência e energia em uma placa real. Simuladores não reproduzem com fidelidade caches, flash externa, periféricos e consumo elétrico.
