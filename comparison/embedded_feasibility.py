#!/usr/bin/env python3
"""Static embedded-memory preflight for a trained PolyGabor model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "polygarbor" / "src"))

from polygarbor import PolyGaborClassifier  # noqa: E402


TARGETS = (
    {
        "target": "ESP32-WROOM-32",
        "architecture": "Xtensa LX6 32-bit, dual-core",
        "clock_mhz": 240,
        "flash_bytes": 4 * 1024 * 1024,
        "sram_bytes": 520 * 1024,
    },
    {
        "target": "Arduino Uno R3",
        "architecture": "ATmega328P AVR 8-bit",
        "clock_mhz": 16,
        "flash_bytes": 32 * 1024,
        "sram_bytes": 2 * 1024,
    },
    {
        "target": "PIC16F877A",
        "architecture": "PIC 8-bit",
        "clock_mhz": 20,
        "flash_bytes": 8192 * 14 // 8,
        "sram_bytes": 368,
    },
)


def array_inventory(classifier: PolyGaborClassifier) -> dict[str, int]:
    floating_elements = 0
    integer_elements = 0
    loaded_actual_bytes = 0
    polynomial_workspace_elements = 0
    for model in classifier.models.values():
        arrays = [model.samples, model.center]
        for level in model.levels:
            arrays.extend(value for value in vars(level).values() if isinstance(value, np.ndarray))
        for array in arrays:
            loaded_actual_bytes += array.nbytes
            if array is model.samples:
                continue
            if np.issubdtype(array.dtype, np.floating):
                floating_elements += array.size
            elif np.issubdtype(array.dtype, np.integer):
                integer_elements += array.size
        for previous, current in zip(model.levels, model.levels[1:]):
            expanded = previous.d_proj * (previous.d_proj + 3) // 2
            selected, projected = current.A_basis.shape
            polynomial_workspace_elements = max(polynomial_workspace_elements, expanded + 2 * selected + 2 * projected)
    return {
        "loaded_actual_bytes": loaded_actual_bytes,
        "floating_elements": floating_elements,
        "integer_elements": integer_elements,
        "float32_uint16_bytes": floating_elements * 4 + integer_elements * 2,
        "int16_uint16_bytes": floating_elements * 2 + integer_elements * 2,
        "int8_uint16_bytes": floating_elements + integer_elements * 2,
        "polynomial_workspace_float32_bytes": polynomial_workspace_elements * 4,
    }


def classify_target(target: dict[str, int | str], measurements: dict[str, int]) -> str:
    flash = int(target["flash_bytes"])
    ram = int(target["sram_bytes"])
    if measurements["int8_state_and_kernels_bytes"] > flash or measurements["streaming_working_set_bytes"] > ram:
        return "inviável para o método atual, mesmo na estimativa quantizada"
    if measurements["fitted_state_and_kernels_bytes"] + 256 * 1024 > flash or measurements["opencv_image_maps_lower_bound_bytes"] > ram:
        return "plausível somente após porte C/C++, quantização e extração em fluxo"
    return "há orçamento estático; ainda requer porte e medição no hardware"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "comparison" / "runs" / "2026-09-12" / "full__polygarbor__seed42" / "model")
    parser.add_argument("--output", type=Path, default=ROOT / "comparison" / "results" / "2026-09-12" / "embedded")
    args = parser.parse_args()

    classifier = PolyGaborClassifier.load(args.model)
    meta = json.loads((args.model / "model.json").read_text())
    artifact_bytes = sum(path.stat().st_size for path in args.model.rglob("*") if path.is_file())
    inventory = array_inventory(classifier)
    height = width = 150
    pixels = height * width
    filters = len(classifier.bank)
    kernel_side = classifier.gabor.ksize
    kernel_elements = filters * 2 * kernel_side * kernel_side
    kernel_float32_bytes = kernel_elements * 4
    kernel_int8_bytes = kernel_elements
    rgb_bytes = pixels * 3
    gray_float32_bytes = pixels * 4
    lab_float32_bytes = pixels * 3 * 4
    energy_float32_bytes = pixels * filters * 4
    streaming_extraction_bytes = kernel_side * width + classifier.n_features * 4 + classifier.n_features * 2 * 8 + classifier.n_classes * 4
    streaming_working_set_bytes = streaming_extraction_bytes + inventory["polynomial_workspace_float32_bytes"]
    measurements = {
        "artifact_bytes": artifact_bytes,
        "training_sample_bank_float32_bytes": sum(samples.nbytes for samples in classifier.train_samples.values()),
        "loaded_runtime_arrays_bytes": inventory["loaded_actual_bytes"],
        "exported_inference_arrays_float32_uint16_bytes": inventory["float32_uint16_bytes"],
        "fitted_state_and_kernels_bytes": inventory["float32_uint16_bytes"] + kernel_float32_bytes,
        "int16_state_and_kernels_bytes": inventory["int16_uint16_bytes"] + kernel_elements * 2,
        "int8_state_and_kernels_bytes": inventory["int8_uint16_bytes"] + kernel_int8_bytes,
        "rgb_input_bytes": rgb_bytes,
        "opencv_image_maps_lower_bound_bytes": rgb_bytes + gray_float32_bytes + lab_float32_bytes + energy_float32_bytes + kernel_float32_bytes,
        "streaming_extraction_bytes": streaming_extraction_bytes,
        "polynomial_workspace_float32_bytes": inventory["polynomial_workspace_float32_bytes"],
        "streaming_working_set_bytes": streaming_working_set_bytes,
        "gabor_direct_multiply_accumulates": pixels * filters * 2 * kernel_side * kernel_side,
        "classes": classifier.n_classes,
        "features": classifier.n_features,
        "filters": filters,
        "kernel_side": kernel_side,
        "samples_per_class": int(next(iter(meta["train_samples_per_class"].values()))),
    }

    rows = []
    for target in TARGETS:
        flash = int(target["flash_bytes"])
        ram = int(target["sram_bytes"])
        rows.append({
            **target,
            "artifact_flash_percent": 100 * artifact_bytes / flash,
            "fitted_state_flash_percent": 100 * measurements["fitted_state_and_kernels_bytes"] / flash,
            "int8_state_flash_percent": 100 * measurements["int8_state_and_kernels_bytes"] / flash,
            "opencv_maps_sram_percent": 100 * measurements["opencv_image_maps_lower_bound_bytes"] / ram,
            "streaming_sram_percent": 100 * streaming_working_set_bytes / ram,
            "artifact_fits_flash": artifact_bytes <= flash,
            "fitted_state_fits_sram": measurements["fitted_state_and_kernels_bytes"] <= ram,
            "int8_state_fits_flash": measurements["int8_state_and_kernels_bytes"] <= flash,
            "opencv_maps_fit_sram": measurements["opencv_image_maps_lower_bound_bytes"] <= ram,
            "streaming_working_set_fits_sram": streaming_working_set_bytes <= ram,
            "training_on_device": False,
            "assessment": classify_target(target, measurements),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(args.model.relative_to(ROOT)),
        "scope": "static memory and operation-count preflight; not instruction-level emulation or hardware timing",
        "measurements": measurements,
        "targets": rows,
        "assumptions": [
            "PIC program capacity is expressed as the bit-equivalent of 8192 words of 14 bits; code also consumes this space.",
            "The loaded-array measurement includes retained training samples; exported inference-state estimates exclude samples because evaluate() does not read them and replace integer indices with uint16.",
            "The int8 estimate changes numeric precision and must be validated against predictions and macro-F1.",
            "The streaming working set assumes row-wise input, fixed-point convolution and online mean/variance; that implementation does not exist yet.",
            "Firmware, stack, heap, image decoder, filesystem and runtime libraries are not included in flash or SRAM percentages.",
        ],
    }
    (args.output / "feasibility.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "feasibility.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Pré-avaliação de inferência em microcontroladores",
        "",
        "Esta análise compara o modelo `full__polygarbor__seed42` com os limites publicados para três placas ou microcontroladores. Ela mede armazenamento e arrays reais do modelo carregado, estima uma representação quantizada e conta operações do banco de Gabor. Ela não executa o firmware do alvo, não prevê latência e não mede energia.",
        "",
        "| Alvo | Flash | SRAM | artefato salvo/flash | estado ajustado/flash | int8/flash | mapas atuais/SRAM | fluxo mínimo/SRAM | avaliação |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['target']} | {row['flash_bytes']:,} B | {row['sram_bytes']:,} B | {row['artifact_flash_percent']:.1f}% | {row['fitted_state_flash_percent']:.1f}% | {row['int8_state_flash_percent']:.1f}% | {row['opencv_maps_sram_percent']:.1f}% | {row['streaming_sram_percent']:.1f}% | {row['assessment']} |")
    lines.extend([
        "",
        f"O diretório salvo ocupa {artifact_bytes / 1_000_000:.3f} MB, mas contém 2.800 vetores de treino em texto. Ao carregar, a biblioteca reconstrói bases polinomiais e mantém as amostras: os arrays ocupam {inventory['loaded_actual_bytes'] / 1_000_000:.3f} MB. As amostras não são consultadas por evaluate() e podem ser excluídas de um exportador. Com kernels, uma exportação de inferência float32 ocuparia pelo menos {measurements['fitted_state_and_kernels_bytes'] / 1_000_000:.3f} MB, antes do firmware. A estimativa int8/uint16 cai para {measurements['int8_state_and_kernels_bytes'] / 1_000_000:.3f} MB, mas sua equivalência preditiva ainda precisa ser testada.",
        "",
        f"A extração atual mantém RGB, cinza float32, Lab float32 e oito mapas de energia float32, um piso de {measurements['opencv_image_maps_lower_bound_bytes'] / 1_000_000:.3f} MB sem contar temporários do OpenCV. Uma implementação em fluxo usaria aproximadamente {streaming_extraction_bytes / 1000:.1f} kB para extração. Somado a um espaço de trabalho float32 conservador de {inventory['polynomial_workspace_float32_bytes'] / 1000:.1f} kB para avaliar uma classe por vez, o total estimado é {streaming_working_set_bytes / 1000:.1f} kB. Esse núcleo precisaria ser escrito em C/C++ com aritmética fixa ou cuidadosamente quantizada.",
        "",
        f"O banco executa duas convoluções 21×21 para cada um dos oito filtros. Uma imagem 150×150 exige aproximadamente {measurements['gabor_direct_multiply_accumulates']:,} multiplicações-acumulações no cálculo direto, além de magnitude, estatísticas Lab e distância polinomial. Por isso, razão de frequências de clock não é uma estimativa válida de latência.",
        "",
        "O ESP32 é o único dos três alvos com uma rota plausível para a inferência completa: exportar o estado ajustado sem reconstrução, quantizar, manter coeficientes na flash e processar a imagem em fluxo. O Arduino Uno e o PIC16F877A não comportam sequer a estimativa int8 do estado polinomial na memória de programa; para eles seria necessário trocar o classificador por uma aproximação muito menor, reduzir filtros e resolução ou usá-los apenas como controladores de um coprocessador.",
        "",
        "Treinar no dispositivo não é uma meta realista nos três alvos. A etapa usa amostras por classe, expansões polinomiais e decomposições numéricas, com memória de trabalho muito acima da inferência. A avaliação embarcada defensável é treinar externamente, exportar uma representação de inferência e medir fidelidade, latência, RAM, flash e energia no alvo.",
        "",
        "As especificações usadas são as folhas oficiais do [ESP32-WROOM-32](https://documentation.espressif.com/esp32-wroom-32_datasheet_en.html), do [Arduino UNO R3](https://docs.arduino.cc/resources/datasheets/A000066-datasheet.pdf) e do [PIC16F87XA](https://ww1.microchip.com/downloads/en/devicedoc/39582b.pdf).",
        "",
        "## Próxima etapa necessária",
        "",
        "A simulação fiel exige um exportador de coeficientes e um núcleo de inferência C/C++. Depois disso, a sequência correta é verificar concordância de classes e degradação de macro-F1/G-mean no teste inteiro, compilar com o toolchain de cada alvo, obter mapa de memória e contagem de ciclos no simulador e medir latência e energia em uma placa real. Simuladores não reproduzem com fidelidade caches, flash externa, periféricos e consumo elétrico.",
        "",
    ])
    (args.output / "README.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "measurements": measurements, "targets": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
