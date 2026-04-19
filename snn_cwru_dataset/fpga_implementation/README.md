# FPGA Implementation: XADC-Driven SNN Edge Accelerator

This directory contains the VHDL hardware implementation of the 1-8-1 Heterogeneous Spiking Neural Network (SNN) developed for real-time mechanical fault detection using the Case Western Reserve University (CWRU) bearing dataset.

Designed as an ultra-low-power, mixed-signal edge accelerator, this architecture bypasses the need for external microcontrollers or external ADCs. By leveraging the integrated **Xilinx XADC**, the FPGA ingests raw, high-frequency analog voltages directly from piezoelectric vibration sensors, performs 0-DSP temporal inference, and drives physical hardware alarms with zero network latency.

## 📁 Hardware Hierarchy

The RTL design is optimized for resource efficiency, utilizing a compact 1-8-1 Leaky Integrate-and-Fire (LIF) topology:

* `snn_cwru_top.vhd`: The mixed-signal top-level wrapper. Instantiates the XADC IP to sample analog input channels and routes the digitized 8-bit/12-bit data directly into the neuromorphic inference engine.
* `neuromorphic_core_1_8_1.vhd`: The temporal inference core. Evaluates the aggressively quantized 1-8-1 LIF network, utilizing independent leakage parameters to isolate acoustic resonance bands specific to bearing faults.
* `snn_weights_pkg.vhd`: The static VHDL package containing the hardware-quantized weight matrices and heterogeneous shift-registers, extracted from the PyTorch Genetic Algorithm training pipeline.

## ⚡ Mixed-Signal Physical Interface (XADC)

Unlike standard digital IP cores, this architecture bridges the analog-digital divide entirely on-chip:

1. **Direct Sensor Ingestion:** The design is configured to interface directly with analog industrial vibration sensors (e.g., Piezoelectric Ceramic Transducers or analog MEMS like the ADXL1002/ADXL335) via the FPGA's auxiliary analog pins (`vauxp`/`vauxn`).
2. **Continuous Conversion:** The XADC is instantiated in continuous sampling mode, safely digitizing the transient, high-frequency acoustic "ringing" of mechanical spalls (sampled at ranges compatible with the 12 kHz CWRU baseline).
3. **Hardware Triggering:** Fault classifications are tied directly to physical output pins (LEDs/Relays), demonstrating a fully closed-loop hardware protection mechanism capable of initiating emergency motor shutdowns within milliseconds of fault detection.

## 📊 Verification & Results

The design includes a robust VHDL testbench environment that replays thousands of baseline and faulted mechanical vibration samples through the RTL simulation to guarantee hardware-software mathematical equivalence.

**Self-Checking Simulation Results:**
*(Insert your screenshot of the Vivado TCL Console showing the CWRU accuracy printouts here)*

**Analog Ingestion & Spiking Waveform:**
The waveform below illustrates the digital translation of the analog input wave, the membrane voltage accumulation inside the 8 hidden neurons, and the ultimate firing of the `master_alarm` upon detecting a physical anomaly.

*(Insert your screenshot of the Vivado Waveform showing the ADC input data and the SNN spike triggers here)*