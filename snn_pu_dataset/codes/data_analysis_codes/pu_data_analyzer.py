# ==============================================================================
# PADERBORN UNIVERSITY (PU) DATASET: AUTOMATED TIME & FREQUENCY ANALYZER
# Generates professional metrics, high-resolution plots, and statistical tables.
# Built for 64 kHz sampling rates and massive 2048-sample observation windows.
# ==============================================================================

import os
import glob
import numpy as np
import scipy.io as sio
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DIRECTORY CONFIGURATION
# ==============================================================================
# Assuming script is run from snn_pu_dataset/codes/data_analysis_codes/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_analysis", "analysis_screenshots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. PADERBORN DATASET CONFIGURATION
# ==============================================================================
class Config:
    FS = 64000            # PU Dataset Sampling Rate (64 kHz)
    ADC_VREF = 5.0        # Assumed ADC Reference Voltage
    WINDOW_SIZE = 2048    # 32 ms observation window (to clear the AM Dead Zone)

# Representative files for Waveform plots (1500 RPM vs 900 RPM comparison)
REPRESENTATIVE_FILES = [
    "N15_M07_F10_K001_1.mat", "N15_M07_F10_KI01_1.mat", "N15_M07_F10_KA01_1.mat",
    "N09_M07_F10_K001_1.mat", "N09_M07_F10_KI01_1.mat", "N09_M07_F10_KA01_1.mat"
]

# ==============================================================================
# 3. METRIC CALCULATION ENGINE
# ==============================================================================
def get_raw_vibration(path):
    """
    Safely navigates the complex nested structs of Paderborn .mat files 
    to extract the raw vibration channel.
    """
    mat = sio.loadmat(path)
    main_key = [k for k in mat.keys() if not k.startswith('__')][0]
    y_array = mat[main_key]['Y'][0, 0]
    
    for i in range(y_array.shape[1]):
        channel = y_array[0, i]
        ch_name_array = channel['Name'][0]
        if len(ch_name_array) > 0 and 'vibration' in str(ch_name_array[0]).lower():
            return channel['Data'].flatten()
    
    # Fallback to index 6 if name matching fails
    return y_array[0, 6]['Data'].flatten()

def calculate_time_domain_metrics(signal):
    # Quantize to 8-bit to simulate FPGA ADC ingestion
    scaled = (np.abs(signal) / Config.ADC_VREF) * 255.0
    quantized = np.clip(scaled, 0, 255).astype(np.int32)
    
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms if rms > 0 else 0
    kurt = kurtosis(signal, fisher=False) # Fisher=False means Normal == 3.0
    skewness = skew(signal)
    return rms, peak, crest_factor, kurt, skewness

# ==============================================================================
# 4. PLOTTING ENGINES
# ==============================================================================
def plot_waveform_and_fft(signal, filename, fault_type, rpm):
    t = np.arange(len(signal)) / Config.FS
    
    # Take a 0.2s slice for a clean time-domain plot at 64kHz
    slice_len = int(0.2 * Config.FS)
    t_slice = t[:slice_len]
    sig_slice = signal[:slice_len]

    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1/Config.FS)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Helicopter Gearbox Signature: {fault_type} ({rpm} RPM)", fontsize=16, fontweight='bold')

    ax1.plot(t_slice, sig_slice, color='#1f77b4', linewidth=0.5)
    ax1.set_title("Time-Domain Waveform (0.2 Seconds)", fontsize=12)
    ax1.set_ylabel("Amplitude (g)")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot up to 10 kHz to capture structural resonance
    ax2.plot(freqs, fft_vals, color='#ff7f0e', linewidth=0.5)
    ax2.set_title("Frequency-Domain Spectrum (FFT)", fontsize=12)
    ax2.set_ylabel("Magnitude")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_xlim(0, 10000) 
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"PU_Signature_{fault_type}_{rpm}RPM.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregate_metrics(metric_dict, metric_name, filename):
    labels = list(metric_dict.keys())
    values = [np.mean(metric_dict[k]) for k in labels]
    errors = [np.std(metric_dict[k]) for k in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, yerr=errors, capsize=5, color=['#2ca02c', '#d62728', '#9467bd'], alpha=0.8, edgecolor='black')
    
    plt.title(f"Average {metric_name} by Fault Type (PU 64kHz Dataset)", fontsize=14, fontweight='bold')
    plt.ylabel(metric_name)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(errors)*0.05), f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 5. STATISTICAL TERMINAL PRINTOUT
# ==============================================================================
def print_statistical_summary(metrics_dict):
    print("\n" + "="*85)
    print(f"{'STATISTICAL SUMMARY OF TIME-DOMAIN METRICS (PU HELICOPTER DATASET)':^85}")
    print("="*85)

    for metric_name, fault_data in metrics_dict.items():
        print(f"\n--- {metric_name.upper()} ---")
        print(f"{'Fault Type':<25} | {'Mean':<12} | {'Std Dev':<12} | {'Min':<12} | {'Max':<12}")
        print("-" * 80)
        for fault_type, values in fault_data.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                print(f"{fault_type:<25} | {mean_val:<12.4f} | {std_val:<12.4f} | {min_val:<12.4f} | {max_val:<12.4f}")
            else:
                print(f"{fault_type:<25} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12}")
    print("\n" + "="*85)

# ==============================================================================
# 6. MAIN ORCHESTRATOR
# ==============================================================================
def run_analysis():
    print(f"\n⚙️ INITIATING PU 64kHz DATA ANALYSIS PIPELINE")
    print("="*60)
    
    metrics_by_type = {
        "RMS (Overall Energy)": {"Normal (Healthy)": [], "Inner Race Fault": [], "Outer Race Fault": []},
        "Peak Amplitude": {"Normal (Healthy)": [], "Inner Race Fault": [], "Outer Race Fault": []},
        "Crest Factor": {"Normal (Healthy)": [], "Inner Race Fault": [], "Outer Race Fault": []},
        "Kurtosis (Spikiness)": {"Normal (Healthy)": [], "Inner Race Fault": [], "Outer Race Fault": []}
    }

    processed_count = 0
    
    # Iterate through the subfolders (k001, kI001, ka001)
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path): continue
            
        all_files = glob.glob(os.path.join(folder_path, "*.mat"))

        for filepath in all_files:
            filename = os.path.basename(filepath)
            
            # Determine Fault Type
            if "K001" in filename: fault_type = "Normal (Healthy)"
            elif "KI01" in filename: fault_type = "Inner Race Fault"
            elif "KA01" in filename: fault_type = "Outer Race Fault"
            else: continue
                
            # Determine RPM
            rpm = 1500 if "N15" in filename else 900 if "N09" in filename else "Unknown"

            try:
                signal = get_raw_vibration(filepath)
            except Exception as e:
                print(f"   [Error] Skipping {filename}: {e}")
                continue

            # 1. Calculate Metrics
            rms, peak, crest, kurt, skewness = calculate_time_domain_metrics(signal)
            
            metrics_by_type["RMS (Overall Energy)"][fault_type].append(rms)
            metrics_by_type["Peak Amplitude"][fault_type].append(peak)
            metrics_by_type["Crest Factor"][fault_type].append(crest)
            metrics_by_type["Kurtosis (Spikiness)"][fault_type].append(kurt)
            
            processed_count += 1

            # 2. Generate Representative Signature Plots
            if filename in REPRESENTATIVE_FILES:
                plot_waveform_and_fft(signal, filename, fault_type, rpm)

    print("\n📊 Generating Aggregate Statistical Bar Charts...")
    plot_aggregate_metrics(metrics_by_type["Kurtosis (Spikiness)"], "Kurtosis", "PU_Aggregate_Kurtosis.png")
    plot_aggregate_metrics(metrics_by_type["RMS (Overall Energy)"], "RMS", "PU_Aggregate_RMS.png")
    plot_aggregate_metrics(metrics_by_type["Peak Amplitude"], "Peak Acceleration (g)", "PU_Aggregate_Peak.png")

    print_statistical_summary(metrics_by_type)

    print(f"✅ Analysis Complete! Processed {processed_count} files.")
    print(f"✅ All high-resolution graphics saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_analysis()