# Ultra-Low Power Edge-AI for Bearing Fault Detection (CWRU Dataset Phase)

## 1. Executive Summary
[cite_start]The digital design and algorithmic training phases of the Spiking Neural Network (SNN) fault detection project are complete[cite: 46]. [cite_start]This repository contains the foundational architecture for an ultra-low-power edge-computing anomaly detection system, specifically targeting bearing faults in rotating machinery[cite: 390]. 

[cite_start]The network has been successfully trained using a hardware-asymmetric frequency curriculum, heavily constrained to an 18-byte memory footprint[cite: 6]. [cite_start]The final VHDL architecture, incorporating a native Artix-7 XADC wrapper, has been synthesized and implemented[cite: 7]. [cite_start]Evaluated on a strictly quarantined zero-shot validation set, the VHDL-synthesized core achieved 100.00% macro-accuracy [cite: 126][cite_start], proving its viability as an ultra-efficient edge AI accelerator[cite: 8].

---

## 2. Hardware-Aware SNN Architecture
[cite_start]To fit within the strict power budgets of edge-AI, traditional Convolutional Neural Networks (CNNs) were abandoned in favor of a Leaky Integrate-and-Fire (LIF) SNN[cite: 147].

* [cite_start]**Topology:** 1-Input, 8-Hidden, 1-Output (1-8-1) LIF network[cite: 253].
* [cite_start]**Multiplier-less Logic:** The neural "leak" was implemented in VHDL using a bitwise arithmetic right-shift (>> 1), completely eliminating the need for DSP blocks[cite: 148].
* [cite_start]**Strict Quantization:** All synaptic weights and membrane potentials were constrained to 8-bit signed integers (-127 to 127)[cite: 149, 181].
* [cite_start]**Mechanical Debouncer:** A post-processing VHDL hardware accumulator evaluates 75 sequential micro-windows (~100 milliseconds)[cite: 208]. [cite_start]It only triggers a Master Alarm if $\ge$ 25 spikes are detected, acting as a flawless noise filter[cite: 151].

---

## 3. Data Forensics & Kinematic Adjustments
[cite_start]Raw integration of the Case Western Reserve University (CWRU) dataset into automated machine learning pipelines presents severe hazards[cite: 129]. [cite_start]A custom data-engineering pipeline was constructed to rectify systemic anomalies[cite: 130]:

* [cite_start]**Sampling Rate Aliasing Prevention:** The 48 kHz baseline files (Files 97, 98, 99, 100) were forensically isolated[cite: 134]. [cite_start]A zero-phase FIR filter safely downsampled the baselines to 12 kHz without phase distortion[cite: 135].
* [cite_start]**Workspace Corruption:** Corrupted data dumps, specifically within File 99 (2 Motor HP Baseline), contained overlapping arrays from File 98[cite: 137, 138]. [cite_start]This rogue data was permanently scrubbed from the training environment[cite: 139].
* [cite_start]**Kinematic Anomalies (The 0.014" Paradox):** The 0.014-inch fault recordings exhibit severe physical anomalies, as documented by Smith & Randall (2015)[cite: 142]. [cite_start]Specifically, the 0.014" Ball Fault repeatedly rolled out of the mechanical load zone, resulting in prolonged periods of acoustic silence[cite: 144]. [cite_start]These physically compromised files were explicitly quarantined to ensure the network learned fundamental physics rather than overfitting to background noise[cite: 145].

---

## 4. The "Hardware-Asymmetric" Training Curriculum
[cite_start]Because a 32-sample window (2.6 ms) is physically too short to observe fundamental kinematic defect frequencies, the network could not learn standard kinematic fault patterns[cite: 246]. 

* [cite_start]**The FFT "Teacher":** During software training (via a PyTorch-accelerated Genetic Algorithm), an FFT extracted high-frequency structural resonance bands (1,500 Hz - 4,500 Hz)[cite: 201]. 
* [cite_start]**Time-Domain Inputs:** This frequency data was never fed to the network's inputs[cite: 249]. [cite_start]The GA's fitness function simply "rewarded" the network when its temporal spikes aligned with resonance events[cite: 250].
* [cite_start]**The 3 HP Stress Test:** The Genetic Algorithm was trained exclusively under the maximum mechanical load condition (3 Motor HP)[cite: 153]. [cite_start]By forcing the SNN to identify fault resonance within the loudest baseline noise floor, the algorithm evolved an incredibly strict threshold gate[cite: 154].

---

## 5. Implementation Metrics & Silicon Proof
[cite_start]The final routed design in Vivado confirms that the system is exceptionally lightweight and stable, perfectly suited for battery-powered industrial environments[cite: 29].

### Target: Xilinx Artix-7 (xc7a35tcpg236-1)
* [cite_start]**Total On-Chip Power:** 74 mW [cite: 30]
* [cite_start]**Dynamic Power (SNN Core):** 2 mW [cite: 31]
* [cite_start]**DSP Blocks Utilized:** 0 [cite: 32]
* [cite_start]**Block RAM (BRAM):** 0 [cite: 301]
* [cite_start]**Worst Negative Slack (WNS):** +70.495 ns (Constrained at 10 MHz) [cite: 34, 76]

---

## 6. Zero-Shot Generalization Results
[cite_start]Evaluated strictly on 58 entirely unseen load and fault variations, the network maintained a 99.80% macro-accuracy, proving it learned the underlying physics rather than memorizing dataset noise[cite: 14].

| Evaluation Metric | Vivado (VHDL Self-Check) |
| :--- | :--- |
| **Total Audio Samples Tested** | [cite_start]192,000 simulated clock cycles [cite: 438] |
| **False Positives (Healthy Alarms)** | [cite_start]0 [cite: 15] |
| **Micro-Window Accuracy (2.6 ms)** | [cite_start]89.17% [cite: 12] |
| **Macro-Window Accuracy (100 ms)** | [cite_start]100.00% (Zero-Shot) [cite: 126] |

[cite_start]*Note: Across the entire cleaned dataset, all false negatives were physically justified by the kinematic load-zone exiting of the rolling elements, which the hardware debouncer correctly mitigated[cite: 17, 18, 19, 20].*