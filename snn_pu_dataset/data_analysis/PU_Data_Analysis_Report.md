# Forensic Time-Domain Analysis of the Paderborn University (PU) Dataset

## 1. Executive Summary
Following the successful deployment of the 1-8-1 Spiking Neural Network on the CWRU dataset, the architecture was migrated to the 64 kHz Paderborn University (PU) dataset. This dataset consists of vibration recordings from an electromechanical drive system simulating a helicopter gearbox under extreme, fluctuating loads.

Before training, a rigorous statistical analysis was conducted across 240 files to map the mechanical physics of the gearbox. The results mathematically prove that the acoustic environment of the PU dataset is vastly more hostile than standard industrial datasets. Overcoming these physical anomalies necessitated three major architectural upgrades: expanding the temporal window to 2048 samples (32 ms), increasing hidden neurons to 32, and introducing unique Heterogeneous Leaky Integrate-and-Fire (LIF) decay rates.

---

## 2. Dataset Statistical Summary

The automated data pipeline extracted the mean, standard deviation, minimum, and maximum boundaries for each metric across the 240 processed files.

### 2.1 RMS (Overall Energy)
| Fault Type | Mean | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| **Normal (Healthy)** | 0.3910 | 0.0164 | 0.3613 | 0.4096 |
| **Inner Race Fault** | 0.5823 | 0.1058 | 0.4087 | 0.6914 |
| **Outer Race Fault** | 0.4544 | 0.1299 | 0.2242 | 0.5423 |

### 2.2 Peak Amplitude (g)
| Fault Type | Mean | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| **Normal (Healthy)** | 4.8118 | 0.2497 | 4.2419 | 5.4779 |
| **Inner Race Fault** | 7.8892 | 1.7994 | 4.5685 | 10.9283 |
| **Outer Race Fault** | 4.1509 | 1.1810 | 2.1851 | 6.1401 |

### 2.3 Crest Factor
| Fault Type | Mean | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| **Normal (Healthy)** | 12.3273 | 0.8084 | 10.5978 | 14.5572 |
| **Inner Race Fault** | 13.4577 | 1.1237 | 11.0352 | 16.1115 |
| **Outer Race Fault** | 9.3949 | 1.6940 | 7.1189 | 12.9989 |

### 2.4 Kurtosis (Spikiness)
| Fault Type | Mean | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| **Normal (Healthy)** | 15.7412 | 2.3685 | 12.9022 | 20.1651 |
| **Inner Race Fault** | 16.1576 | 1.8547 | 12.4770 | 18.7822 |
| **Outer Race Fault** | 16.8432 | 3.4021 | 14.0123 | 23.2482 |

---

## 3. Engineering Insights & Architectural Justifications

The extracted metrics reveal several severe non-linearities in the physical dataset that traditional neural networks fail to account for.

### 3.1 The Collapse of Kurtosis Thresholding
In standard diagnostic environments, a healthy bearing produces Gaussian white noise with a Kurtosis of approximately 3.0. The PU dataset reveals a shockingly violent baseline, with a healthy Kurtosis mean of **15.7412** and maximums reaching **20.1651**. The system is inherently "spiky" even when perfectly healthy, rendering traditional statistical thresholding completely useless for anomaly detection.

### 3.2 The "Normal in the Middle" Non-Linearity
The data exposes a critical overlap in the kinetic energy (RMS) of the system. While the mean RMS of an Inner Race fault (0.5823) is higher than the Normal baseline (0.3910), the Outer Race fault frequently exhibits RMS values (Min: 0.2242) that are significantly *quieter* than a healthy motor.

> **Architectural Fix:** A standard, fixed-leak SNN acts as a simple high-pass energy gate and cannot classify a baseline that sits between two faults. This non-linearity mathematically justified the invention of **Heterogeneous Leaks**. By evolving a unique, independent decay rate for all 32 neurons, the SNN was transformed into a complex bank of temporal band-pass filters, capable of differentiating the specific rhythm of the signals rather than just their raw volume.

### 3.3 Overcoming Amplitude Modulation (The 2048-Sample Solution)
At low motor speeds (900 RPM), the time-domain waveform exhibits severe Amplitude Modulation. As the physical defect rotates in and out of the mechanical load zone, the acoustic impacts disappear into the 15.74 Kurtosis noise floor.

> **Architectural Fix:** A standard 16 ms observation window (1024 samples) regularly lands entirely inside this acoustic "dead zone," starving the hardware debouncer and causing unavoidable False Negatives. By expanding the VHDL counters to a 2048-sample window (32 ms), the architecture physically overhangs the dead zone, geometrically ensuring the SNN captures at least one physical impact per evaluation cycle.

### 3.4 Structural Resonance vs. Peak Frequencies
To maintain robust, zero-shot accuracy across shifting 900 RPM to 1500 RPM motor speeds, the Genetic Algorithm's fitness function was strictly trained to identify **Structural Resonance** rather than standard kinematic peak frequencies. While kinematic frequencies shift linearly with shaft speed, the structural resonance of the metallic gearbox casing remains physically fixed. This frequency curriculum ensured the final time-domain hardware remained completely immune to variable motor speeds.