# =====================================================================
# MULTI-DOMAIN MASS FORENSIC EVALUATION BENCHMARK
# Testing 480 files across 4 Operating Conditions (2 Seen, 2 Unseen)
# Automatically adapts to Google Colab or Local VS Code environments.
# =====================================================================

import os
import numpy as np
import scipy.io as sio
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running Multi-Domain Forensic Hardware Benchmark on: {device}")

# =====================================================================
# 1. AUTOMATED GENOME LOADER
# =====================================================================
HIDDEN_FEATURES = 32

W1_ARRAY = [np.int32(46), np.int32(-25), np.int32(47), np.int32(-16), np.int32(46), np.int32(-86), np.int32(-66), np.int32(52), np.int32(118), np.int32(22), np.int32(-36), np.int32(78), np.int32(-96), np.int32(-73), np.int32(8), np.int32(-15), np.int32(-17), np.int32(109), np.int32(-22), np.int32(16), np.int32(-4), np.int32(28), np.int32(127), np.int32(-66), np.int32(15), np.int32(81), np.int32(111), np.int32(-103), np.int32(59), np.int32(-22), np.int32(-52), np.int32(-72)]
W2_ARRAY = [np.int32(127), np.int32(-62), np.int32(-88), np.int32(-127), np.int32(-99), np.int32(-11), np.int32(95), np.int32(88), np.int32(10), np.int32(3), np.int32(47), np.int32(4), np.int32(116), np.int32(127), np.int32(80), np.int32(-51), np.int32(-100), np.int32(-66), np.int32(73), np.int32(-33), np.int32(-3), np.int32(-8), np.int32(127), np.int32(74), np.int32(-39), np.int32(127), np.int32(118), np.int32(-106), np.int32(50), np.int32(-74), np.int32(20), np.int32(49)]
LEAK_ARRAY = [np.int32(3), np.int32(3), np.int32(2), np.int32(4), np.int32(3), np.int32(0), np.int32(2), np.int32(2), np.int32(0), np.int32(2), np.int32(0), np.int32(1), np.int32(0), np.int32(0), np.int32(3), np.int32(2), np.int32(1), np.int32(2), np.int32(1), np.int32(4), np.int32(2), np.int32(1), np.int32(3), np.int32(1), np.int32(1), np.int32(2), np.int32(2), np.int32(0), np.int32(2), np.int32(0), np.int32(2), np.int32(0)]
T1_VAL = 2925
T2_VAL = 709

# =====================================================================
# 2. CONFIGURATION & DYNAMIC DATA PIPELINE
# =====================================================================
WINDOW_SIZE = 2048
MACRO_WINDOW_SIZE = 12
ADC_VREF = 5.0

# 🟢 DYNAMIC PATH DETECTOR (Solves VS Code vs Colab issue)
if os.path.exists("/content"):
    # We are running inside Google Colab
    print("☁️ Google Colab Environment Detected.")
    DATA_DIR = "/content/drive/MyDrive/data" # Ensure your Drive is mounted!
else:
    # We are running locally in VS Code
    print("💻 Local VS Code Environment Detected.")
    # Automatically finds the 'data' folder one level up from the 'codes' folder
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")

def extract_vibration_data(filepath):
    mat = sio.loadmat(filepath)
    main_keys = [k for k in mat.keys() if not k.startswith('__')]
    main_key = main_keys[0]
    try:
        y_array = mat[main_key]['Y'][0, 0]
        for i in range(y_array.shape[1]):
            channel = y_array[0, i]
            ch_name_array = channel['Name'][0]
            if len(ch_name_array) > 0 and 'vibration' in str(ch_name_array[0]).lower():
                return channel['Data'].flatten()
        return y_array[0, 6]['Data'].flatten()
    except Exception as e:
        raise ValueError(f"Failed to extract from {filepath}: {e}")

def create_windows(data, window_size):
    num_windows = len(data) // window_size
    return data[:num_windows * window_size].reshape(num_windows, window_size)

def quantize_to_8bit_adc(raw_signal):
    scaled = (np.abs(raw_signal) / ADC_VREF) * 255.0
    return np.clip(scaled, 0, 255).astype(np.int32)

# =====================================================================
# 3. SILICON INFERENCE ENGINE
# =====================================================================
def run_hardware_inference(X_raw):
    POP, NUM_WINDOWS = 1, X_raw.shape[0]
    if NUM_WINDOWS == 0: return np.array([])

    mem1 = torch.zeros((POP, NUM_WINDOWS, HIDDEN_FEATURES), dtype=torch.int32, device=device)
    mem2 = torch.zeros((POP, NUM_WINDOWS), dtype=torch.int32, device=device)
    micro_alarms = torch.zeros((POP, NUM_WINDOWS), dtype=torch.int32, device=device)

    w1_exp = torch.tensor(W1_ARRAY, dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(1)
    w2_exp = torch.tensor(W2_ARRAY, dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(1)
    leaks_exp = torch.tensor(LEAK_ARRAY, dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(1)
    t1_exp = torch.tensor([T1_VAL], dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(2)
    t2_exp = torch.tensor([T2_VAL], dtype=torch.int32, device=device).unsqueeze(0)

    X_tensor = torch.tensor(X_raw, dtype=torch.int32, device=device).unsqueeze(0)

    for step in range(WINDOW_SIZE):
        x_t_exp = X_tensor[:, :, step].unsqueeze(2)
        cur1 = x_t_exp * w1_exp
        mem1 = (mem1 >> leaks_exp) + cur1
        spk1 = (mem1 > t1_exp).to(torch.int32)
        mem1 = mem1 * (1 - spk1)
        cur2 = torch.sum(spk1 * w2_exp, dim=2)
        mem2 = (mem2 >> 1) + cur2
        spk2 = (mem2 > t2_exp).to(torch.int32)
        mem2 = mem2 * (1 - spk2)
        micro_alarms |= spk2

    return micro_alarms[0].cpu().numpy()

def evaluate_macro_blocks(preds, is_fault):
    TP = TN = FP = FN = 0
    num_macros = len(preds) // MACRO_WINDOW_SIZE
    alarm_threshold = max(1, MACRO_WINDOW_SIZE // 2)

    for i in range(num_macros):
        macro_preds = preds[i * MACRO_WINDOW_SIZE : (i+1) * MACRO_WINDOW_SIZE]
        alarmed = 1 if np.sum(macro_preds) >= alarm_threshold else 0
        if alarmed == 1 and is_fault == 1: TP += 1
        elif alarmed == 0 and is_fault == 0: TN += 1
        elif alarmed == 1 and is_fault == 0: FP += 1
        elif alarmed == 0 and is_fault == 1: FN += 1
    return TP, TN, FP, FN

def evaluate_micro_blocks(preds, is_fault):
    TP = TN = FP = FN = 0
    if is_fault == 1:
        TP = np.sum(preds == 1)
        FN = np.sum(preds == 0)
    else:
        TN = np.sum(preds == 0)
        FP = np.sum(preds == 1)
    return TP, TN, FP, FN

# =====================================================================
# 4. THE MASS EVALUATION
# =====================================================================
conditions = [
    {"code": "N15_M07_F10", "name": "SEEN TRAINING DATA (1500 RPM Base)"},
    {"code": "N09_M07_F10", "name": "SEEN TRAINING DATA (900 RPM Base)"},
    {"code": "N15_M01_F10", "name": "UNSEEN ZERO-SHOT (0.1 Nm Low Torque Drop)"},
    {"code": "N15_M07_F04", "name": "UNSEEN ZERO-SHOT (400 N Low Radial Force Drop)"}
]

classes = [
    {"code": "K001", "is_fault": 0, "name": "Healthy Baseline"},
    {"code": "KI01", "is_fault": 1, "name": "Artificial Inner Race"},
    {"code": "KA01", "is_fault": 1, "name": "Artificial Outer Race"},
    {"code": "KI04", "is_fault": 1, "name": "REAL Natural Inner Race"}, 
    {"code": "KB23", "is_fault": 1, "name": "REAL Natural Inner + Outer Race"}, 
    {"code": "KA04", "is_fault": 1, "name": "REAL Natural Outer Race"}  
]

results = {cond["code"]: {cls["code"]: {
    "macro_TP":0, "macro_TN":0, "macro_FP":0, "macro_FN":0,
    "micro_TP":0, "micro_TN":0, "micro_FP":0, "micro_FN":0
} for cls in classes} for cond in conditions}

print("\n" + "="*70)
print("⚙️ INITIATING MASSIVE MULTI-DOMAIN FORENSIC BENCHMARK")
print(f"📁 Target Data Directory: {DATA_DIR}")
print("="*70)

start_time = time.time()

for cond in conditions:
    print(f"\n📂 CATEGORY: {cond['name']}")
    print("-" * 70)

    cond_macro_TP = cond_macro_TN = cond_macro_FP = cond_macro_FN = 0
    cond_micro_TP = cond_micro_TN = cond_micro_FP = cond_micro_FN = 0

    for cls in classes:
        X_class_list = []
        files_found = 0
        class_dir_path = os.path.join(DATA_DIR, cls['code'])

        for i in range(1, 21):
            filename = f"{cond['code']}_{cls['code']}_{i}.mat"
            filepath = os.path.join(class_dir_path, filename)

            if os.path.exists(filepath):
                try:
                    raw = extract_vibration_data(filepath)
                    wins = create_windows(quantize_to_8bit_adc(raw), WINDOW_SIZE)
                    X_class_list.append(wins)
                    files_found += 1
                except Exception:
                    pass

        if files_found > 0:
            X_class = np.concatenate(X_class_list, axis=0)
            preds = run_hardware_inference(X_class)

            macro_TP, macro_TN, macro_FP, macro_FN = evaluate_macro_blocks(preds, cls['is_fault'])
            micro_TP, micro_TN, micro_FP, micro_FN = evaluate_micro_blocks(preds, cls['is_fault'])

            results[cond['code']][cls['code']] = {
                "macro_TP": macro_TP, "macro_TN": macro_TN, "macro_FP": macro_FP, "macro_FN": macro_FN,
                "micro_TP": micro_TP, "micro_TN": micro_TN, "micro_FP": micro_FP, "micro_FN": micro_FN
            }
            cond_macro_TP += macro_TP; cond_macro_TN += macro_TN; cond_macro_FP += macro_FP; cond_macro_FN += macro_FN
            cond_micro_TP += micro_TP; cond_micro_TN += micro_TN; cond_micro_FP += micro_FP; cond_micro_FN += micro_FN

            macro_acc = (macro_TP+macro_TN)/(macro_TP+macro_TN+macro_FP+macro_FN) * 100 if (macro_TP+macro_TN+macro_FP+macro_FN) > 0 else 0
            micro_acc = (micro_TP+micro_TN)/(micro_TP+micro_TN+micro_FP+micro_FN) * 100 if (micro_TP+micro_TN+micro_FP+micro_FN) > 0 else 0
            print(f"   [{cls['code']} - {cls['name']}] Macro Acc: {macro_acc:6.2f}% | Micro Acc: {micro_acc:6.2f}% | TP:{macro_TP:4} | TN:{macro_TN:4} | FP:{macro_FP:4} | FN:{macro_FN:4} | (Files: {files_found})")
        else:
            print(f"   [{cls['code']} - {cls['name']}] ❌ NO FILES FOUND")

    c_macro_acc = (cond_macro_TP+cond_macro_TN)/(cond_macro_TP+cond_macro_TN+cond_macro_FP+cond_macro_FN) * 100 if (cond_macro_TP+cond_macro_TN+cond_macro_FP+cond_macro_FN) > 0 else 0
    c_micro_acc = (cond_micro_TP+cond_micro_TN)/(cond_micro_TP+cond_micro_TN+cond_micro_FP+cond_micro_FN) * 100 if (cond_micro_TP+cond_micro_TN+cond_micro_FP+cond_micro_FN) > 0 else 0

    print(f"   >>> CONDITION AGGREGATE MACRO ACCURACY: {c_macro_acc:.2f}%")
    print(f"   >>> CONDITION AGGREGATE MICRO ACCURACY: {c_micro_acc:.2f}%\n")

# =====================================================================
# 5. GRAND AGGREGATION & REPORTING
# =====================================================================
def sum_metrics(cond_list, cls_list=None):
    if cls_list is None: cls_list = [c["code"] for c in classes]
    macro_TP = macro_TN = macro_FP = macro_FN = 0
    micro_TP = micro_TN = micro_FP = micro_FN = 0
    for c in cond_list:
        for cl in cls_list:
            m = results[c][cl]
            macro_TP+=m["macro_TP"]; macro_TN+=m["macro_TN"]; macro_FP+=m["macro_FP"]; macro_FN+=m["macro_FN"]
            micro_TP+=m["micro_TP"]; micro_TN+=m["micro_TN"]; micro_FP+=m["micro_FP"]; micro_FN+=m["micro_FN"]
    return (macro_TP, macro_TN, macro_FP, macro_FN), (micro_TP, micro_TN, micro_FP, micro_FN)

def calc_acc(TP, TN, FP, FN):
    return (TP+TN)/(TP+TN+FP+FN) * 100 if (TP+TN+FP+FN) > 0 else 0

seen_conds = ["N15_M07_F10", "N09_M07_F10"]
unseen_conds = ["N15_M01_F10", "N15_M07_F04"]
all_conds = seen_conds + unseen_conds

(TP_all, TN_all, FP_all, FN_all), (m_TP_all, m_TN_all, m_FP_all, m_FN_all) = sum_metrics(all_conds)
(TP_seen, TN_seen, FP_seen, FN_seen), (m_TP_seen, m_TN_seen, m_FP_seen, m_FN_seen) = sum_metrics(seen_conds)
(TP_uns, TN_uns, FP_uns, FN_uns), (m_TP_uns, m_TN_uns, m_FP_uns, m_FN_uns) = sum_metrics(unseen_conds)

# 🟢 THE FIX: Safely aggregating Artificial + Real Damages into the final printout
(TP_in, TN_in, FP_in, FN_in), (m_TP_in, m_TN_in, m_FP_in, m_FN_in) = sum_metrics(all_conds, ["KI01", "KI04"])
(TP_out, TN_out, FP_out, FN_out), (m_TP_out, m_TN_out, m_FP_out, m_FN_out) = sum_metrics(all_conds, ["KA01", "KA04"])
(TP_combo, TN_combo, FP_combo, FN_combo), (m_TP_combo, m_TN_combo, m_FP_combo, m_FN_combo) = sum_metrics(all_conds, ["KB23"])
(TP_h, TN_h, FP_h, FN_h), (m_TP_h, m_TN_h, m_FP_h, m_FN_h) = sum_metrics(all_conds, ["K001"])

print("="*70)
print("🏆 GRAND AGGREGATE REPORT (MULTI-DOMAIN)")
print("="*70)

print(f"1. OVERALL MACRO ACCURACY          : {calc_acc(TP_all, TN_all, FP_all, FN_all):6.2f}% (TP:{TP_all} TN:{TN_all} FP:{FP_all} FN:{FN_all})")
print(f"   OVERALL MICRO ACCURACY          : {calc_acc(m_TP_all, m_TN_all, m_FP_all, m_FN_all):6.2f}% (TP:{m_TP_all} TN:{m_TN_all} FP:{m_FP_all} FN:{m_FN_all})")
print(f"2. SEEN DATA (1500+900) MACRO      : {calc_acc(TP_seen, TN_seen, FP_seen, FN_seen):6.2f}% (TP:{TP_seen} TN:{TN_seen} FP:{FP_seen} FN:{FN_seen})")
print(f"   SEEN DATA (1500+900) MICRO      : {calc_acc(m_TP_seen, m_TN_seen, m_FP_seen, m_FN_seen):6.2f}% (TP:{m_TP_seen} TN:{m_TN_seen} FP:{m_FP_seen} FN:{m_FN_seen})")
print(f"3. UNSEEN DATA (Torque/Force) MACRO: {calc_acc(TP_uns, TN_uns, FP_uns, FN_uns):6.2f}% (TP:{TP_uns} TN:{TN_uns} FP:{FP_uns} FN:{FN_uns})")
print(f"   UNSEEN DATA (Torque/Force) MICRO: {calc_acc(m_TP_uns, m_TN_uns, m_FP_uns, m_FN_uns):6.2f}% (TP:{m_TP_uns} TN:{m_TN_uns} FP:{m_FP_uns} FN:{m_FN_uns})")

print("\n--- FAULT SPECIFIC BREAKDOWN (Includes Artificial + Real) ---")
print(f"4. TOTAL INNER RACE MACRO ACCURACY : {calc_acc(TP_in, TN_in, FP_in, FN_in):6.2f}% (TP:{TP_in} FN:{FN_in})")
print(f"   TOTAL INNER RACE MICRO ACCURACY : {calc_acc(m_TP_in, m_TN_in, m_FP_in, m_FN_in):6.2f}% (TP:{m_TP_in} FN:{m_FN_in})")
print(f"5. TOTAL OUTER RACE MACRO ACCURACY : {calc_acc(TP_out, TN_out, FP_out, FN_out):6.2f}% (TP:{TP_out} FN:{FN_out})")
print(f"   TOTAL OUTER RACE MICRO ACCURACY : {calc_acc(m_TP_out, m_TN_out, m_FP_out, m_FN_out):6.2f}% (TP:{m_TP_out} FN:{m_FN_out})")
print(f"6. TOTAL COMBO (KB23) MACRO ACC    : {calc_acc(TP_combo, TN_combo, FP_combo, FN_combo):6.2f}% (TP:{TP_combo} FN:{FN_combo})")
print(f"   TOTAL COMBO (KB23) MICRO ACC    : {calc_acc(m_TP_combo, m_TN_combo, m_FP_combo, m_FN_combo):6.2f}% (TP:{m_TP_combo} FN:{m_FN_combo})")
print(f"7. TOTAL HEALTHY MACRO SPECIFICITY : {calc_acc(TP_h, TN_h, FP_h, FN_h):6.2f}% (TN:{TN_h} FP:{FP_h})  <-- False Alarms")
print(f"   TOTAL HEALTHY MICRO SPECIFICITY : {calc_acc(m_TP_h, m_TN_h, m_FP_h, m_FN_h):6.2f}% (TN:{m_TN_h} FP:{m_FP_h})  <-- False Alarms")

print("\n--- CONDITION SPECIFIC BREAKDOWN ---")
(t,n,p,f), (mt,mn,mp,mf) = sum_metrics(["N15_M07_F10"]); print(f"8. 1500 RPM (Seen) MACRO           : {calc_acc(t,n,p,f):6.2f}%")
print(f"   1500 RPM (Seen) MICRO           : {calc_acc(mt,mn,mp,mf):6.2f}%")
(t,n,p,f), (mt,mn,mp,mf) = sum_metrics(["N09_M07_F10"]); print(f"9. 900 RPM (Seen) MACRO            : {calc_acc(t,n,p,f):6.2f}%")
print(f"   900 RPM (Seen) MICRO            : {calc_acc(mt,mn,mp,mf):6.2f}%")
(t,n,p,f), (mt,mn,mp,mf) = sum_metrics(["N15_M01_F10"]); print(f"10. 0.1 Nm Torque (Unseen) MACRO   : {calc_acc(t,n,p,f):6.2f}%")
print(f"   0.1 Nm Torque (Unseen) MICRO    : {calc_acc(mt,mn,mp,mf):6.2f}%")
(t,n,p,f), (mt,mn,mp,mf) = sum_metrics(["N15_M07_F04"]); print(f"11. 400 N Force (Unseen) MACRO     : {calc_acc(t,n,p,f):6.2f}%")
print(f"   400 N Force (Unseen) MICRO      : {calc_acc(mt,mn,mp,mf):6.2f}%")

print(f"\n⏱️ Total Simulation Time: {int(time.time() - start_time)} seconds")
print("="*70)