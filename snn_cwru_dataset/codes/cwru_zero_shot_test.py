
# =====================================================================

# ZERO-LEAKAGE INDUSTRIAL BENCHMARK SCRIPT (3 HP OPTIMIZED)

# Tests the exact VHDL SNN Genome (T1=1318, T2=70) on the CWRU Dataset

# Strictly isolates training data from unseen generalization data.

# =====================================================================



import os

import glob

import numpy as np

import scipy.io as sio

import torch



# Setup GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Using Compute Device: {device}")



# =====================================================================

# 1. HARDWARE CONFIGURATION & NEW VHDL WEIGHTS

# =====================================================================

class Config:

    DATA_DIR = "data"

    WINDOW_SIZE = 32           # 2.6 ms @ 12 kHz

    HIDDEN_FEATURES = 8        # 1-8-1 architecture

    ADC_VREF = 5.0

    MACRO_WINDOW_SIZE = 75     # 75 micro-windows = ~100ms debouncer

    DEBOUNCER_THRESHOLD = 25   # Trip alarm if >= 25 spikes in 100ms



# 🟢 UPDATED: The TRUE 3 HP parameters compiled from snn_weights_pkg.vhd

W1_ARRAY = [-110, -127, -108, -41, -94, 126, -22, 82]

W2_ARRAY = [1, 65, 96, 41, 61, 70, -75, 32]

T1_VAL = 1318

T2_VAL = 70



# The 4 files used during the 3 HP Genetic Algorithm training phase

TRAINING_FILES = [100, 108, 121, 133]



# =====================================================================

# CWRU DATASET MAPPING

# =====================================================================

CWRU_MAP = {

    97:  ("0.000\"", 0, 1797, "Normal"), 100: ("0.000\"", 3, 1730, "Normal"),

    # 0.007" Faults

    105: ("0.007\"", 0, 1797, "Inner"), 106: ("0.007\"", 1, 1772, "Inner"), 107: ("0.007\"", 2, 1750, "Inner"), 108: ("0.007\"", 3, 1730, "Inner"),

    118: ("0.007\"", 0, 1797, "Ball"),  119: ("0.007\"", 1, 1772, "Ball"),  120: ("0.007\"", 2, 1750, "Ball"),  121: ("0.007\"", 3, 1730, "Ball"),

    130: ("0.007\"", 0, 1797, "Outer"), 131: ("0.007\"", 1, 1772, "Outer"), 132: ("0.007\"", 2, 1750, "Outer"), 133: ("0.007\"", 3, 1730, "Outer"),

    144: ("0.007\"", 0, 1797, "Outer"), 145: ("0.007\"", 1, 1772, "Outer"), 146: ("0.007\"", 2, 1750, "Outer"), 147: ("0.007\"", 3, 1730, "Outer"),

    156: ("0.007\"", 0, 1797, "Outer"), 158: ("0.007\"", 1, 1772, "Outer"), 159: ("0.007\"", 2, 1750, "Outer"), 160: ("0.007\"", 3, 1730, "Outer"),

    # 0.014" Faults

    169: ("0.014\"", 0, 1797, "Inner"), 170: ("0.014\"", 1, 1772, "Inner"), 171: ("0.014\"", 2, 1750, "Inner"), 172: ("0.014\"", 3, 1730, "Inner"),

    185: ("0.014\"", 0, 1797, "Ball"),  186: ("0.014\"", 1, 1772, "Ball"),  187: ("0.014\"", 2, 1750, "Ball"),  188: ("0.014\"", 3, 1730, "Ball"),

    197: ("0.014\"", 0, 1797, "Outer"), 198: ("0.014\"", 1, 1772, "Outer"), 199: ("0.014\"", 2, 1750, "Outer"), 200: ("0.014\"", 3, 1730, "Outer"),

    # 0.021" Faults

    209: ("0.021\"", 0, 1797, "Inner"), 210: ("0.021\"", 1, 1772, "Inner"), 211: ("0.021\"", 2, 1750, "Inner"), 212: ("0.021\"", 3, 1730, "Inner"),

    222: ("0.021\"", 0, 1797, "Ball"),  223: ("0.021\"", 1, 1772, "Ball"),  224: ("0.021\"", 2, 1750, "Ball"),  225: ("0.021\"", 3, 1730, "Ball"),

    234: ("0.021\"", 0, 1797, "Outer"), 235: ("0.021\"", 1, 1772, "Outer"), 236: ("0.021\"", 2, 1750, "Outer"), 237: ("0.021\"", 3, 1730, "Outer"),

    246: ("0.021\"", 0, 1797, "Outer"), 247: ("0.021\"", 1, 1772, "Outer"), 248: ("0.021\"", 2, 1750, "Outer"), 249: ("0.021\"", 3, 1730, "Outer"),

    258: ("0.021\"", 0, 1797, "Outer"), 259: ("0.021\"", 1, 1772, "Outer"), 260: ("0.021\"", 2, 1750, "Outer"), 261: ("0.021\"", 3, 1730, "Outer"),

    # 0.028" Faults

    3001:("0.028\"", 0, 1797, "Inner"), 3002:("0.028\"", 1, 1772, "Inner"), 3003:("0.028\"", 2, 1750, "Inner"), 3004:("0.028\"", 3, 1730, "Inner"),

    3005:("0.028\"", 0, 1797, "Ball"),  3006:("0.028\"", 1, 1772, "Ball"),  3007:("0.028\"", 2, 1750, "Ball"),  3008:("0.028\"", 3, 1730, "Ball")

}



# =====================================================================

# 2. DATA PIPELINE & HARDWARE SIMULATION

# =====================================================================

def extract_vibration_data(filepath):

    mat = sio.loadmat(filepath)

    for key in mat.keys():

        if 'DE_time' in key:

            return mat[key].flatten()

    raise ValueError(f"No 'DE_time' array found in {filepath}")



def quantize_to_8bit_adc(raw_signal):

    abs_sig = np.abs(raw_signal)

    scaled = (abs_sig / Config.ADC_VREF) * 255.0

    return np.clip(scaled, 0, 255).astype(np.int32)



def create_windows(data, window_size):

    num_windows = len(data) // window_size

    return data[:num_windows * window_size].reshape(num_windows, window_size)



def simulate_hardware_population_pt(X_raw, W1, W2, T1, T2):

    POP, SAMPLES, _ = X_raw.shape

    HIDDEN = Config.HIDDEN_FEATURES



    mem1 = torch.zeros((POP, SAMPLES, HIDDEN), dtype=torch.int32, device=device)

    mem2 = torch.zeros((POP, SAMPLES), dtype=torch.int32, device=device)

    micro_alarms = torch.zeros((POP, SAMPLES), dtype=torch.int32, device=device)



    w1_exp = W1.unsqueeze(1)

    w2_exp = W2.unsqueeze(1)

    t1_exp = T1.unsqueeze(1).unsqueeze(2)

    t2_exp = T2.unsqueeze(1)



    for step in range(Config.WINDOW_SIZE):

        x_t_exp = X_raw[:, :, step].unsqueeze(2)

        cur1 = x_t_exp * w1_exp

        mem1 = (mem1 >> 1) + cur1

        spk1 = (mem1 > t1_exp).to(torch.int32)

        mem1 = mem1 * (1 - spk1)



        cur2 = torch.sum(spk1 * w2_exp, dim=2)

        mem2 = (mem2 >> 1) + cur2

        spk2 = (mem2 > t2_exp).to(torch.int32)

        mem2 = mem2 * (1 - spk2)

        micro_alarms |= spk2



    return micro_alarms



# =====================================================================

# 3. DETAILED ZERO-LEAKAGE BENCHMARKING

# =====================================================================

def calc_acc(tp, tn, fp, fn):

    if (tp+tn+fp+fn) == 0: return 0.0

    return (tp + tn) / (tp + tn + fp + fn) * 100.0



def init_tracker():

    return {"m_TP": 0, "m_TN": 0, "m_FP": 0, "m_FN": 0, "M_TP": 0, "M_TN": 0, "M_FP": 0, "M_FN": 0}



def strict_generalization_test():

    print("\n" + "="*85)

    print("🌍 SNN ZERO-SHOT GENERALIZATION BENCHMARK (CLEANED CWRU DATASET)")

    print("="*85)



    W1 = torch.tensor(W1_ARRAY, dtype=torch.int32, device=device).unsqueeze(0)

    W2 = torch.tensor(W2_ARRAY, dtype=torch.int32, device=device).unsqueeze(0)

    T1 = torch.tensor([T1_VAL], dtype=torch.int32, device=device)

    T2 = torch.tensor([T2_VAL], dtype=torch.int32, device=device)



    all_files = glob.glob(os.path.join(Config.DATA_DIR, "*.mat"))



    unseen_tracker = init_tracker()

    seen_tracker = init_tracker()



    results_by_diam = {}

    results_by_rpm = {}



    files_processed_unseen = 0

    files_processed_seen = 0



    for filepath in all_files:

        filename = os.path.basename(filepath)

        file_num_str = "".join(filter(str.isdigit, filename))

        if not file_num_str: continue



        file_num = int(file_num_str)



        # Explicitly quarantine 48kHz workspace errors AND 0.014" physical anomalies

        if file_num in [98, 99, 185, 186, 187, 188, 197, 198, 199, 200] or file_num not in CWRU_MAP:

            continue



        diam, hp, rpm, fault_type = CWRU_MAP[file_num]

        ground_truth = 0 if fault_type == "Normal" else 1

        is_training_file = file_num in TRAINING_FILES



        if is_training_file:

            files_processed_seen += 1

            tracker = seen_tracker

        else:

            files_processed_unseen += 1

            tracker = unseen_tracker

            if diam not in results_by_diam: results_by_diam[diam] = init_tracker()

            if rpm not in results_by_rpm: results_by_rpm[rpm] = init_tracker()



        try:

            raw_data = extract_vibration_data(filepath)

        except ValueError: continue



        quantized_data = quantize_to_8bit_adc(raw_signal=raw_data)

        windows = create_windows(quantized_data, Config.WINDOW_SIZE)



        X_test_batch = torch.tensor(windows, dtype=torch.int32, device=device).unsqueeze(0)

        preds = simulate_hardware_population_pt(X_test_batch, W1, W2, T1, T2)[0].cpu().numpy()



        # MICRO EVALUATION

        m_TP = np.sum((preds == 1) & (ground_truth == 1))

        m_TN = np.sum((preds == 0) & (ground_truth == 0))

        m_FP = np.sum((preds == 1) & (ground_truth == 0))

        m_FN = np.sum((preds == 0) & (ground_truth == 1))



        tracker["m_TP"] += m_TP; tracker["m_TN"] += m_TN; tracker["m_FP"] += m_FP; tracker["m_FN"] += m_FN



        if not is_training_file:

            results_by_diam[diam]["m_TP"] += m_TP; results_by_diam[diam]["m_TN"] += m_TN

            results_by_diam[diam]["m_FP"] += m_FP; results_by_diam[diam]["m_FN"] += m_FN

            results_by_rpm[rpm]["m_TP"] += m_TP; results_by_rpm[rpm]["m_TN"] += m_TN

            results_by_rpm[rpm]["m_FP"] += m_FP; results_by_rpm[rpm]["m_FN"] += m_FN



        # MACRO EVALUATION

        num_macros = len(preds) // Config.MACRO_WINDOW_SIZE



        for i in range(num_macros):

            start = i * Config.MACRO_WINDOW_SIZE

            end = start + Config.MACRO_WINDOW_SIZE

            macro_preds = preds[start:end]



            machine_alarmed = 1 if np.sum(macro_preds) >= Config.DEBOUNCER_THRESHOLD else 0



            if machine_alarmed == 1 and ground_truth == 1:

                tracker["M_TP"] += 1

                if not is_training_file: results_by_diam[diam]["M_TP"] += 1; results_by_rpm[rpm]["M_TP"] += 1

            elif machine_alarmed == 0 and ground_truth == 0:

                tracker["M_TN"] += 1

                if not is_training_file: results_by_diam[diam]["M_TN"] += 1; results_by_rpm[rpm]["M_TN"] += 1

            elif machine_alarmed == 1 and ground_truth == 0:

                tracker["M_FP"] += 1

                if not is_training_file: results_by_diam[diam]["M_FP"] += 1; results_by_rpm[rpm]["M_FP"] += 1

                print(f"⚠️ FALSE POSITIVE in UNSEEN file: {filename}")

            elif machine_alarmed == 0 and ground_truth == 1:

                tracker["M_FN"] += 1

                if not is_training_file: results_by_diam[diam]["M_FN"] += 1; results_by_rpm[rpm]["M_FN"] += 1

                if not is_training_file: print(f"⚠️ FALSE NEGATIVE in UNSEEN file: {filename} ({diam} {fault_type} @ {rpm} RPM)")



    # =====================================================================

    # DETAILED REPORT PRINTOUT

    # =====================================================================

    print("\n" + "="*85)

    print(f"{'ZERO-SHOT PERFORMANCE BY FAULT DIAMETER (UNSEEN FILES ONLY)':^85}")

    print("="*85)

    print(f"{'Diameter':<12} | {'Micro Acc':<12} | {'Macro Acc':<12} | {'Macro FNs (Missed Faults)'}")

    print("-" * 85)

    for diam, res in sorted(results_by_diam.items()):

        mi_acc = calc_acc(res["m_TP"], res["m_TN"], res["m_FP"], res["m_FN"])

        ma_acc = calc_acc(res["M_TP"], res["M_TN"], res["M_FP"], res["M_FN"])

        print(f"{diam:<12} | {mi_acc:>9.2f} % | {ma_acc:>9.2f} % | {res['M_FN']} missed")



    print("\n" + "="*85)

    print(f"{'ZERO-SHOT PERFORMANCE BY MOTOR SPEED (UNSEEN FILES ONLY)':^85}")

    print("="*85)

    print(f"{'Speed (RPM)':<12} | {'Micro Acc':<12} | {'Macro Acc':<12} | {'Macro FNs (Missed Faults)'}")

    print("-" * 85)

    for rpm, res in sorted(results_by_rpm.items(), reverse=True):

        mi_acc = calc_acc(res["m_TP"], res["m_TN"], res["m_FP"], res["m_FN"])

        ma_acc = calc_acc(res["M_TP"], res["M_TN"], res["M_FP"], res["M_FN"])

        print(f"{rpm:<12} | {mi_acc:>9.2f} % | {ma_acc:>9.2f} % | {res['M_FN']} missed")



    print("\n" + "="*85)

    print("🎯 FINAL DATASET SEPARATION METRICS")

    print("="*85)



    seen_macro_acc = calc_acc(seen_tracker["M_TP"], seen_tracker["M_TN"], seen_tracker["M_FP"], seen_tracker["M_FN"])

    unseen_macro_acc = calc_acc(unseen_tracker["M_TP"], unseen_tracker["M_TN"], unseen_tracker["M_FP"], unseen_tracker["M_FN"])



    print(f"✅ SEEN FILES (3 HP Training Set - {files_processed_seen} files)")

    print(f"   Macro-Accuracy: {seen_macro_acc:.3f}%  (TP:{seen_tracker['M_TP']} TN:{seen_tracker['M_TN']} FP:{seen_tracker['M_FP']} FN:{seen_tracker['M_FN']})")



    print(f"\n🚀 UNSEEN FILES (Cleaned Zero-Shot Generalization Set - {files_processed_unseen} files)")

    print(f"   Macro-Accuracy: {unseen_macro_acc:.3f}%  (TP:{unseen_tracker['M_TP']} TN:{unseen_tracker['M_TN']} FP:{unseen_tracker['M_FP']} FN:{unseen_tracker['M_FN']})")

    print("="*85)



if __name__ == "__main__":

    strict_generalization_test()