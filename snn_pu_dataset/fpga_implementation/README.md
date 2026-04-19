# FPGA Implementation: Cognitive Sensor Node for Aerospace Telemetry

This directory contains the VHDL implementation of the 1-32-1 Heterogeneous Spiking Neural Network (SNN) developed for real-time kinematic anomaly detection using the PU Dataset. 

Designed with extreme SWaP (Size, Weight, and Power) constraints in mind, this architecture operates as a 10-milliwatt, 0-DSP cognitive coprocessor. It directly ingests raw analog sensor data, classifies temporal mechanical anomalies (such as bearing spalls), and formats the output for immediate transmission over standard avionics telemetry buses.

## 📁 Hardware Hierarchy

The RTL design is entirely modular, completely decoupling the neuromorphic mathematics from the telemetry state machines:

* `snn_avionics_top.vhd`: The top-level wrapper. Acts as the event manager, detecting SNN anomaly spikes and triggering the telemetry packet transmission.
* `neuromorphic_core.vhd`: The inference engine. Evaluates the 1-32-1 LIF (Leaky Integrate-and-Fire) topology with 32 unique heterogeneous leak parameters and a 12-window macro-relay debouncer.
* `uart_tx.vhd`: A lightweight, 0-DSP serial transmitter operating at 115200 baud for Hardware-in-the-Loop (HIL) testing.
* `snn_weights_pkg.vhd`: The hardware-quantized weight matrix extracted directly from the PyTorch Genetic Algorithm.

## 🚀 Avionics Deployment Architecture

This IP core is designed to bridge the gap between edge-AI inference and standard aerospace communication protocols.

### 1. Lab Demonstration & HIL Testing (Current Implementation)
For physical bench testing and live demonstration, the `snn_avionics_top` routes the critical alarm signal (`master_alarm`) through the integrated `uart_tx` module. The moment a fault is classified, the FPGA transmits a `0xFF` telemetry packet over a point-to-point serial link to a ground control laptop with near-zero latency.

### 2. Production UAV Deployment (Target Architecture)
For final deployment within a UAV or missile system, the point-to-point UART module is designed to be bypassed. The modular `master_alarm` pin can be routed directly into a Xilinx AXI CAN Bus IP Core. This allows the SNN to act as a silent, invisible safety node, broadcasting high-priority kinematic alarms (e.g., ID `0x100`) directly to the primary flight controller via the vehicle's shared CAN network.

## 📊 Verification & Results

The design includes a fully automated, self-checking VHDL testbench (`tb_neuromorphic_core.vhd`) that replays 24 seconds of live mechanical data (nearly 1.5 million samples) through the RTL simulation.

**Self-Checking Simulation Results:**
* **Micro-Window Accuracy (32.0ms):** 89.58% 
* **Macro-Window Debouncer Accuracy (384ms):** 100%

*(Insert your screenshot of the TCL Console printout here)*

**Telemetry Verification:**
The waveform below demonstrates the exact millisecond the neuromorphic core detects an Inner Race fault and triggers the UART state machine to transmit the `0xFF` critical alarm packet.

*(Insert your screenshot of the Vivado Waveform showing the UART transmission here)*