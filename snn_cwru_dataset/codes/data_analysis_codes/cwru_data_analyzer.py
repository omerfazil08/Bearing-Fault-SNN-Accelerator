# ==============================================================================
# CWRU DATASET: AUTOMATED TIME-DOMAIN & FREQUENCY-DOMAIN ANALYZER
# Generates professional metrics and high-resolution plots for GitHub documentation.
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
# Assuming script is run from snn_cwru_dataset/codes/data_analysis_codes/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_analysis", "analysis_screenshots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. CWRU DATASET MAPPING (Excluding Known Corrupted Files 98, 99)
# ==============================================================================
CWRU_MAP = {
    97:  ("0.000", 0, "Normal"), 100: ("0.000", 3, "Normal"),
    105: ("0.007", 0, "Inner"), 108: ("0.007", 3, "Inner"),
    118: ("0.007", 0, "Ball"),  121: ("0.007", 3, "Ball"),
    130: ("0.007", 0, "Outer"), 133: ("0.007", 3, "Outer"),
    169: ("0.014", 0, "Inner"), 172: ("0.014", 3, "Inner"),
    197: ("0.014", 0, "Outer"), 200: ("0.014", 3, "Outer"),
    209: ("0.021", 0, "Inner"), 212: ("0.021", 3, "Inner"),
    222: ("0.021", 0, "Ball"),  225: ("0.021", 3, "Ball"),
    234: ("0.021", 0, "Outer"), 237: ("0.021", 3, "Outer"),
    3001:("0.028", 0, "Inner"), 3004:("0.028", 3, "Inner"),
    3005:("0.028", 0, "Ball"),  3008:("0.028", 3, "Ball")
}

# Representative files for the 3 HP Stress Test Waveform plots
REPRESENTATIVE_FILES = [100, 108, 121, 133] 

# ==============================================================================
# 3. METRIC CALCULATION ENGINE
# ==============================================================================
def extract_vibration_data(filepath):
    mat = sio.loadmat(filepath)
    for key in mat.keys():
        if 'DE_time' in key:
            return mat[key].flatten()
    raise ValueError(f"No 'DE_time' array found in {filepath}")

def calculate_time_domain_metrics(signal):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms if rms > 0 else 0
    kurt = kurtosis(signal, fisher=False) # Fisher=False means Normal == 3.0
    skewness = skew(signal)
    return rms, peak, crest_factor, kurt, skewness

# ==============================================================================
# 4. PLOTTING ENGINES
# ==============================================================================
def plot_waveform_and_fft(signal, file_num, fault_type, diam, hp):
    fs = 12000
    t = np.arange(len(signal)) / fs
    
    # Take a 0.1s slice for a clean time-domain plot
    slice_len = int(0.1 * fs)
    t_slice = t[:slice_len]
    sig_slice = signal[:slice_len]

    # Calculate FFT
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Acoustic Signature: {fault_type} Baseline (Diam: {diam}\", Load: {hp} HP)", fontsize=16, fontweight='bold')

    # Time Domain Plot
    ax1.plot(t_slice, sig_slice, color='#1f77b4', linewidth=1)
    ax1.set_title("Time-Domain Waveform (0.1 Seconds)", fontsize=12)
    ax1.set_ylabel("Amplitude (g)")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Frequency Domain Plot
    ax2.plot(freqs, fft_vals, color='#ff7f0e', linewidth=1)
    ax2.set_title("Frequency-Domain Spectrum (FFT)", fontsize=12)
    ax2.set_ylabel("Magnitude")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_xlim(0, 6000)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"Signature_{fault_type}_{file_num}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [Plot Generated] {save_path}")

def plot_aggregate_metrics(metric_dict, metric_name, filename):
    labels = list(metric_dict.keys())
    values = [np.mean(metric_dict[k]) for k in labels]
    errors = [np.std(metric_dict[k]) for k in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, yerr=errors, capsize=5, color=['#2ca02c', '#d62728', '#9467bd', '#8c564b'], alpha=0.8, edgecolor='black')
    
    plt.title(f"Average {metric_name} by Fault Type (Full CWRU Dataset)", fontsize=14, fontweight='bold')
    plt.ylabel(metric_name)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(errors)*0.1), f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [Aggregate Plot Generated] {save_path}")

# ==============================================================================
# 5. MAIN ORCHESTRATOR
# ==============================================================================
def run_analysis():
    print(f"\n⚙️ INITIATING CWRU DATA ANALYSIS PIPELINE")
    print("="*60)
    
    all_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    
    metrics_by_type = {
        "Kurtosis": {"Normal": [], "Inner": [], "Outer": [], "Ball": []},
        "RMS": {"Normal": [], "Inner": [], "Outer": [], "Ball": []},
        "Crest Factor": {"Normal": [], "Inner": [], "Outer": [], "Ball": []}
    }

    processed_count = 0

    for filepath in all_files:
        filename = os.path.basename(filepath)
        file_num_str = "".join(filter(str.isdigit, filename))
        if not file_num_str: continue
        
        file_num = int(file_num_str)
        if file_num not in CWRU_MAP: continue

        diam, hp, fault_type = CWRU_MAP[file_num]
        
        try:
            signal = extract_vibration_data(filepath)
        except Exception as e:
            print(f"   [Error] Skipping {filename}: {e}")
            continue

        # 1. Calculate Metrics
        rms, peak, crest, kurt, skewness = calculate_time_domain_metrics(signal)
        
        metrics_by_type["Kurtosis"][fault_type].append(kurt)
        metrics_by_type["RMS"][fault_type].append(rms)
        metrics_by_type["Crest Factor"][fault_type].append(crest)
        
        processed_count += 1

        # 2. Generate Representative Signature Plots
        if file_num in REPRESENTATIVE_FILES:
            plot_waveform_and_fft(signal, file_num, fault_type, diam, hp)

    print("\n📊 Generating Aggregate Statistical Bar Charts...")
    plot_aggregate_metrics(metrics_by_type["Kurtosis"], "Kurtosis (Spikiness)", "Aggregate_Kurtosis.png")
    plot_aggregate_metrics(metrics_by_type["RMS"], "Root Mean Square (Overall Energy)", "Aggregate_RMS.png")
    plot_aggregate_metrics(metrics_by_type["Crest Factor"], "Crest Factor", "Aggregate_Crest_Factor.png")

    print("="*60)
    print(f"✅ Analysis Complete! Processed {processed_count} files.")
    print(f"✅ All high-resolution graphics saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_analysis()