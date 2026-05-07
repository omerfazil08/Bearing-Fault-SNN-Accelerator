# Bearing‑Fault SNN Accelerator

**Ultra‑low‑power Spiking Neural Network for bearing fault detection on FPGA**

> A hardware‑ready, integer‑only spiking neural network trained with evolutionary algorithms to detect bearing faults from raw vibration data. No DSP blocks, no floating‑point operations – pure VHDL‑friendly logic.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![VHDL](https://img.shields.io/badge/VHDL-2008-green)]()
[![License](https://img.shields.io/badge/License-Custom%20(All%20Rights%20Reserved)-red)](LICENSE)

---

## 🔍 Overview

Industrial bearings are critical components in rotating machinery. Undetected faults cause catastrophic failures, costly downtime, and safety hazards. Vibration monitoring is the gold standard, but existing solutions rely on power‑hungry DSP pipelines or cloud connectivity – impractical for remote, battery‑operated sensors.

This project presents a **spiking neural network (SNN)** that:
- Accepts **raw 8‑bit vibration samples** (no FFT, no feature extraction)
- Runs entirely with **integer arithmetic and bit‑shift operations**
- Is trained with a **genetic algorithm + surrogate‑gradient fine‑tuning** pipeline
- Achieves competitive fault‑detection accuracy on the **CWRU** and **PU** bearing datasets
- Is **fully synthesizable** into VHDL for deployment on AMD/Xilinx FPGAs

---

## 🧠 Architecture

- **Input:** 248 raw samples (15.5 ms @ 16 kHz), no FFT, no filters
- **Hidden layer:** 64 Leaky‑Integrate‑and‑Fire (LIF) neurons with bit‑shift leakage
- **Output:** binary spike that is aggregated over multiple macro‑windows
- **Decision:** an M‑of‑N coincidence detector suppresses spurious false alarms

---

## Repository Structure
Bearing-Fault-SNN-Accelerator/
├── README.md ← you are here
├── requirements.txt ← Python dependencies
├── LICENSE
│
├── src/ ← Training & evaluation
│ ├── train.py ← Main training script (GA + fine‑tune)
│ ├── evaluate.py ← Macro‑window evaluation & sweeps
│ ├── model.py ← PyTorch SNN model + STE quantisation
│ └── utils.py ← Data loading, feature extraction
│
├── vhdl/ ← Synthesizable VHDL
│ ├── neuron.vhd ← Single LIF neuron
│ ├── layer.vhd ← Fully‑connected SNN layer
│ ├── macro_detector.vhd ← M‑of‑N coincidence state machine
│ └── vivado_project.tcl ← Vivado project creation script
│
├── data/ ← Dataset loaders & pre‑processing
│ ├── cwru_loader.py
│ └── pu_loader.py
│
├── results/ ← Saved genomes & evaluation logs
│ ├── best_genome_cwru.npy
│ ├── best_genome_pu.npy
│ └── macro_stats.json
│
└── notebooks/ ← Jupyter tutorials
└── quick_start.ipynb


---

## Datasets

Two publicly available bearing vibration datasets are used to validate the approach:

1. **CWRU Bearing Data Centre** – single point defects, 12 kHz drive‑end accelerometer  
2. **PU (Paderborn University) Bearing Dataset** – both artificial and accelerated‑life real damage, 64 kHz

Both are resampled to 16 kHz and split into 248‑sample windows with an 80/10/10 train/validation/test split.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/omerfazil08/Bearing-Fault-SNN-Accelerator.git
cd Bearing-Fault-SNN-Accelerator