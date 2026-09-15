# Microcontroller inference preflight

This analysis compares the `full__polygarbor__seed42` model with the published limits of three boards or microcontrollers. It measures the storage and the actual arrays of the loaded model, estimates a quantized representation and counts the operations of the Gabor bank. It does not run the target firmware, does not predict latency and does not measure energy.

| Target | Flash | SRAM | saved artifact/flash | fitted state/flash | int8/flash | current maps/SRAM | minimal streaming/SRAM | assessment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ESP32-WROOM-32 | 4,194,304 B | 532,480 B | 15.5% | 85.5% | 21.7% | 220.8% | 5.1% | plausible only after a C/C++ port, quantization and streaming extraction |
| Arduino Uno R3 | 32,768 B | 2,048 B | 1986.1% | 10947.7% | 2775.9% | 57408.4% | 1325.7% | not feasible for the current method, even with the quantized estimate |
| PIC16F877A | 14,336 B | 368 B | 4539.8% | 25023.3% | 6344.9% | 319490.2% | 7377.7% | not feasible for the current method, even with the quantized estimate |

The saved directory takes 0.651 MB, but it contains 2,800 training vectors stored as text. On loading, the library rebuilds the polynomial bases and keeps the samples: the arrays take 3.857 MB. The samples are not read by evaluate() and can be left out by an exporter. With the kernels, a float32 inference export would take at least 3.587 MB, before firmware. The int8/uint16 estimate drops to 0.910 MB, but its predictive equivalence still needs to be tested.

The current extraction keeps RGB, float32 grayscale, float32 Lab and eight float32 energy maps, a floor of 1.176 MB without counting OpenCV temporaries. A streaming implementation would use approximately 3.6 kB for extraction. Adding a conservative float32 workspace of 23.5 kB to evaluate one class at a time, the estimated total is 27.1 kB. That core would have to be written in C/C++ with fixed-point or carefully quantized arithmetic.

The bank runs two 21×21 convolutions for each of the eight filters. A 150×150 image requires approximately 158,760,000 multiply-accumulate operations in the direct computation, plus magnitude, Lab statistics and polynomial distance. For this reason, the ratio of clock frequencies is not a valid latency estimate.

The ESP32 is the only one of the three targets with a plausible route to full inference: export the fitted state without reconstruction, quantize, keep the coefficients in flash and process the image as a stream. The Arduino Uno and the PIC16F877A cannot hold even the int8 estimate of the polynomial state in program memory; for them the classifier would have to be replaced by a much smaller approximation, the filters and resolution reduced, or the boards used only as controllers of a coprocessor.

On-device training is not a realistic goal on any of the three targets. That stage uses per-class samples, polynomial expansions and numerical decompositions, with working memory far above that of inference. The defensible embedded evaluation is to train externally, export an inference representation and measure fidelity, latency, RAM, flash and energy on the target.

The specifications used are the official datasheets of the [ESP32-WROOM-32](https://documentation.espressif.com/esp32-wroom-32_datasheet_en.html), the [Arduino UNO R3](https://docs.arduino.cc/resources/datasheets/A000066-datasheet.pdf) and the [PIC16F87XA](https://ww1.microchip.com/downloads/en/devicedoc/39582b.pdf).

## Required next step

A faithful simulation requires a coefficient exporter and a C/C++ inference core. After that, the right sequence is to check class agreement and macro-F1/G-mean degradation on the whole test set, compile with each target's toolchain, obtain the memory map and cycle count in the simulator, and measure latency and energy on a real board. Simulators do not faithfully reproduce caches, external flash, peripherals or power consumption.
