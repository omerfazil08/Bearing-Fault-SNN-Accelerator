# =====================================================================
# CWRU 1-8-1 SNN NEUROEVOLUTION TRAINER (3 HP STRESS TEST)
# For bearing fault detection using raw time-domain ADC windows.
# No input injection – pure SNN simulation mirroring VHDL logic.
# Accelerated using PyTorch for Google Colab / Local GPU.
# =====================================================================

import os
import time
import numpy as np
import scipy.io as sio
import torch

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Setup GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Compute Device: {device}")

# =====================================================================
# 1. HARDWARE CONFIGURATION
# =====================================================================
class Config:
    DATA_DIR = "../data" # Path adjusted for GitHub structure
    # 3 HP Stress Test Files
    DATA_FILES = {
        0: {"file": "100.mat", "name": "normal"},
        1: {"file": "108.mat", "name": "inner_race"},
        2: {"file": "121.mat", "name": "ball"},
        3: {"file": "133.mat", "name": "outer_race"}
    }

    WINDOW_SIZE = 32           # time samples per window (2.6 ms @ 12 kHz)
    HIDDEN_FEATURES = 8        # 1-8-1 architecture
    ADC_VREF = 5.0
    MACRO_WINDOW_SIZE = 75

    # Genetic Algorithm Parameters
    POP_SIZE = 700
    GENERATIONS = 1000
    MUTATION_RATE = 0.05

    # Curriculum & Fitness Shaping
    RAW_ONLY_GENERATIONS = 30          # first 30 gens: no bonus
    BONUS_RAMP_START = 30              # after gen 30, start adding bonus
    BONUS_RAMP_END = 80                # max bonus at gen 80
    MAX_BONUS_WEIGHT = 0.5             # maximum added fitness bonus
    PERF_THRESHOLD = 0.7               # per-fault recall target
    ADAPTIVE_UPDATE_INTERVAL = 10      # update class weights every N gens

    # High-Frequency Resonance Bands (Tuned for physical ring frequencies)
    FAULT_BANDS = {
        1: (0.15, 0.25),   # inner race
        2: (0.25, 0.35),   # ball
        3: (0.35, 0.45)    # outer race
    }

# =====================================================================
# 2. DATA PIPELINE (Raw ADC windows + multi-class labels)
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

def extract_crucial_features(window_8bit, fault_bands):
    voltage = window_8bit / 255.0 * Config.ADC_VREF
    fft = np.abs(np.fft.rfft(voltage))
    freqs = np.fft.rfftfreq(Config.WINDOW_SIZE)
    features = []
    for (low, high) in fault_bands.values():
        mask = (freqs >= low) & (freqs < high)
        band_energy = np.sum(fft[mask] ** 2)
        features.append(band_energy)
    return np.array(features, dtype=np.float32)

def build_datasets():
    print("📊 Building datasets (raw windows + crucial features)...")
    normal_path = os.path.join(Config.DATA_DIR, Config.DATA_FILES[0]["file"])
    normal_raw = extract_vibration_data(normal_path)

    t_bound = int(len(normal_raw) * 0.70)
    v_bound = int(len(normal_raw) * 0.85)

    X_train_list, Y_bin_list, Y_multi_list = [], [], []
    X_test_list, Y_test_multi_list = [], []

    # Normal
    n_train_wins = create_windows(quantize_to_8bit_adc(normal_raw[:t_bound]), Config.WINDOW_SIZE)
    n_test_wins  = create_windows(quantize_to_8bit_adc(normal_raw[v_bound:]), Config.WINDOW_SIZE)
    X_train_list.append(n_train_wins)
    Y_bin_list.append(np.zeros(len(n_train_wins), dtype=np.int32))
    Y_multi_list.append(np.zeros(len(n_train_wins), dtype=np.int32))
    X_test_list.append(n_test_wins)
    Y_test_multi_list.append(np.zeros(len(n_test_wins), dtype=np.int32))

    for class_id in [1,2,3]:
        fpath = os.path.join(Config.DATA_DIR, Config.DATA_FILES[class_id]["file"])
        f_raw = extract_vibration_data(fpath)
        f_wins = create_windows(quantize_to_8bit_adc(f_raw), Config.WINDOW_SIZE)

        ft = int(len(f_wins) * 0.70)
        fv = int(len(f_wins) * 0.85)
        X_train_list.append(f_wins[:ft])
        Y_bin_list.append(np.ones(ft, dtype=np.int32))
        Y_multi_list.append(np.full(ft, class_id, dtype=np.int32))
        X_test_list.append(f_wins[fv:])
        Y_test_multi_list.append(np.full(len(f_wins)-fv, class_id, dtype=np.int32))

    X_train = np.concatenate(X_train_list, axis=0)
    Y_train_binary = np.concatenate(Y_bin_list, axis=0)
    Y_train_multi = np.concatenate(Y_multi_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    Y_test_multi = np.concatenate(Y_test_multi_list, axis=0)
    Y_test_binary = (Y_test_multi > 0).astype(np.int32)

    # Subsample training for GA speed
    idx = np.random.choice(len(X_train), size=2500, replace=False)
    X_train = X_train[idx]
    Y_train_binary = Y_train_binary[idx]
    Y_train_multi = Y_train_multi[idx]

    print("📈 Computing crucial features (FFT Resonance Bands)...")
    X_train_crucial = np.zeros((len(X_train), len(Config.FAULT_BANDS)), dtype=np.float32)
    for i, win in enumerate(X_train):
        X_train_crucial[i] = extract_crucial_features(win, Config.FAULT_BANDS)

    X_test_crucial = np.zeros((len(X_test), len(Config.FAULT_BANDS)), dtype=np.float32)
    for i, win in enumerate(X_test):
        X_test_crucial[i] = extract_crucial_features(win, Config.FAULT_BANDS)

    return (X_train, Y_train_binary, Y_train_multi, X_train_crucial), \
           (X_test, Y_test_binary, Y_test_multi, X_test_crucial)

# =====================================================================
# 3. GPU/PYTORCH HARDWARE SIMULATION
# =====================================================================
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
# 4. FULLY VECTORIZED GPU FITNESS SHAPING
# =====================================================================
def compute_balanced_accuracy_pt(preds, Y_true):
    Y_true_exp = Y_true.unsqueeze(0)
    TP = torch.sum((preds == 1) & (Y_true_exp == 1), dim=1).float()
    TN = torch.sum((preds == 0) & (Y_true_exp == 0), dim=1).float()
    FP = torch.sum((preds == 1) & (Y_true_exp == 0), dim=1).float()
    FN = torch.sum((preds == 0) & (Y_true_exp == 1), dim=1).float()
    sens = TP / (TP + FN + 1e-6)
    spec = TN / (TN + FP + 1e-6)
    return ((sens + spec) / 2.0).cpu().numpy()

def compute_knowledge_bonus_pt(preds, X_crucial, band_thresholds, class_weights, bonus_weight):
    fault_present = (X_crucial > band_thresholds).to(torch.int32)
    preds_exp = preds.unsqueeze(2)
    fault_exp = fault_present.unsqueeze(0)
    correct_by_band = torch.sum(preds_exp & fault_exp, dim=1).float()
    total_alarms = torch.sum(preds, dim=1).float()
    total_alarms = torch.where(total_alarms == 0, torch.ones_like(total_alarms), total_alarms)
    weighted_precision = torch.sum(correct_by_band * class_weights, dim=1) / total_alarms
    return (bonus_weight * weighted_precision).cpu().numpy()

# =====================================================================
# 5. GENETIC ALGORITHM
# =====================================================================
def run_genetic_algorithm(X_train_raw, Y_train_binary, Y_train_multi, X_train_crucial):
    print(f"\n🧬 Evolving on {device} with Curriculum + Fitness Shaping")

    n_val = int(len(X_train_raw) * 0.1)
    indices = np.random.permutation(len(X_train_raw))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train_t = torch.tensor(X_train_raw[train_idx], dtype=torch.int32, device=device)
    Y_train_bin_t = torch.tensor(Y_train_binary[train_idx], dtype=torch.int32, device=device)
    X_crucial_t = torch.tensor(X_train_crucial[train_idx], dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_train_raw[val_idx], dtype=torch.int32, device=device)
    Y_val_multi_t = torch.tensor(Y_train_multi[val_idx], dtype=torch.int32, device=device)

    X_train_batch = X_train_t.unsqueeze(0).expand(Config.POP_SIZE, -1, -1)
    X_val_batch = X_val_t.unsqueeze(0).expand(1, -1, -1)

    healthy_mask = (Y_train_multi[train_idx] == 0)
    if np.sum(healthy_mask) > 0:
        band_thresh_cpu = np.percentile(X_train_crucial[train_idx][healthy_mask], 90, axis=0)
    else:
        band_thresh_cpu = np.percentile(X_train_crucial[train_idx], 90, axis=0)

    band_thresh_t = torch.tensor(band_thresh_cpu, dtype=torch.float32, device=device)

    n_bands = len(Config.FAULT_BANDS)
    class_weights_t = torch.ones(n_bands, dtype=torch.float32, device=device) / n_bands

    population = np.zeros((Config.POP_SIZE, 18), dtype=np.int32)
    population[:, 0:16] = np.random.randint(-127, 127, size=(Config.POP_SIZE, 16))
    population[:, 16] = np.random.randint(1000, 20000, size=Config.POP_SIZE)
    population[:, 17] = np.random.randint(10, 1000, size=Config.POP_SIZE)

    best_chromosome = None
    best_fitness = 0.0

    for gen in range(Config.GENERATIONS):
        if gen < Config.RAW_ONLY_GENERATIONS:
            bonus_weight = 0.0
        else:
            ramp_progress = (gen - Config.RAW_ONLY_GENERATIONS) / (Config.BONUS_RAMP_END - Config.RAW_ONLY_GENERATIONS)
            bonus_weight = Config.MAX_BONUS_WEIGHT * np.clip(ramp_progress, 0.0, 1.0)

        W1 = torch.tensor(population[:, 0:Config.HIDDEN_FEATURES], dtype=torch.int32, device=device)
        W2 = torch.tensor(population[:, Config.HIDDEN_FEATURES:16], dtype=torch.int32, device=device)
        T1 = torch.tensor(population[:, 16], dtype=torch.int32, device=device)
        T2 = torch.tensor(population[:, 17], dtype=torch.int32, device=device)

        preds = simulate_hardware_population_pt(X_train_batch, W1, W2, T1, T2)
        base_fitness = compute_balanced_accuracy_pt(preds, Y_train_bin_t)

        if bonus_weight > 0:
            bonus = compute_knowledge_bonus_pt(preds, X_crucial_t, band_thresh_t, class_weights_t, bonus_weight)
        else:
            bonus = np.zeros(Config.POP_SIZE)

        total_fitness = base_fitness + bonus
        sorted_idx = np.argsort(total_fitness)[::-1]
        population = population[sorted_idx]
        total_fitness = total_fitness[sorted_idx]
        base_fitness = base_fitness[sorted_idx]
        bonus = bonus[sorted_idx]

        if total_fitness[0] > best_fitness:
            best_fitness = total_fitness[0]
            best_chromosome = population[0].copy()

        if gen >= Config.RAW_ONLY_GENERATIONS and (gen % Config.ADAPTIVE_UPDATE_INTERVAL == 0):
            W1_best = torch.tensor(best_chromosome[0:Config.HIDDEN_FEATURES], dtype=torch.int32, device=device).unsqueeze(0)
            W2_best = torch.tensor(best_chromosome[Config.HIDDEN_FEATURES:16], dtype=torch.int32, device=device).unsqueeze(0)
            T1_best = torch.tensor([best_chromosome[16]], dtype=torch.int32, device=device)
            T2_best = torch.tensor([best_chromosome[17]], dtype=torch.int32, device=device)

            val_preds = simulate_hardware_population_pt(X_val_batch, W1_best, W2_best, T1_best, T2_best)[0]
            recalls = {}
            new_weights = np.ones(n_bands)
            for i, cid in enumerate([1,2,3]):
                mask = (Y_val_multi_t == cid)
                if torch.sum(mask) == 0:
                    recalls[cid] = 1.0
                else:
                    tp = torch.sum(val_preds[mask] == 1).float()
                    fn = torch.sum(val_preds[mask] == 0).float()
                    recalls[cid] = (tp / (tp + fn + 1e-6)).item()

                if recalls[cid] < Config.PERF_THRESHOLD:
                    new_weights[i] = 1.0 + (Config.PERF_THRESHOLD - recalls[cid])
                else:
                    new_weights[i] = 0.5

            class_weights_t = torch.tensor(new_weights / np.sum(new_weights), dtype=torch.float32, device=device)
            print(f"   Gen {gen}: Updated Class Weights = {class_weights_t.cpu().numpy().round(3)}")

        if gen % 10 == 0:
            print(f"Gen {gen:03d} | Best Fit: {total_fitness[0]:.4f} (base={base_fitness[0]:.3f}, bonus={bonus[0]:.3f}) | T1={population[0,16]}, T2={population[0,17]}")

        next_gen = np.zeros_like(population)
        elite_count = int(Config.POP_SIZE * 0.1)
        next_gen[:elite_count] = population[:elite_count]

        mut_var_w = max(5, int(127 * (1.0 - (gen / Config.GENERATIONS))))
        mut_var_t1 = max(500, int(5000 * (1.0 - (gen / Config.GENERATIONS))))
        mut_var_t2 = max(20, int(200 * (1.0 - (gen / Config.GENERATIONS))))

        for i in range(elite_count, int(Config.POP_SIZE * 0.8)):
            p1 = population[np.random.randint(0, elite_count * 2)]
            p2 = population[np.random.randint(0, elite_count * 2)]
            mask = np.random.randint(0, 2, size=18).astype(bool)
            child = np.where(mask, p1, p2)
            if np.random.rand() < Config.MUTATION_RATE:
                child[0:16] += np.random.randint(-mut_var_w, mut_var_w, size=16)
                child[16] += np.random.randint(-mut_var_t1, mut_var_t1)
                child[17] += np.random.randint(-mut_var_t2, mut_var_t2)
            child[0:16] = np.clip(child[0:16], -127, 127)
            child[16] = np.clip(child[16], 500, 32000)
            child[17] = np.clip(child[17], 5, 2000)
            next_gen[i] = child

        random_start = int(Config.POP_SIZE * 0.8)
        next_gen[random_start:, 0:16] = np.random.randint(-127, 127, size=(Config.POP_SIZE - random_start, 16))
        next_gen[random_start:, 16] = np.random.randint(1000, 20000, size=Config.POP_SIZE - random_start)
        next_gen[random_start:, 17] = np.random.randint(10, 1000, size=Config.POP_SIZE - random_start)
        population = next_gen

    print(f"✅ Evolution Complete. Best Fitness: {best_fitness:.4f}")
    return best_chromosome

# =====================================================================
# 6. CPU-BASED EVALUATION (For Final Testing)
# =====================================================================
def evaluate_macro_window(preds, targets, macro_size):
    print("\n" + "="*50)
    print("🏆 FINAL EVALUATION (MACRO-WINDOW / ~100ms)")
    print("="*50)
    TP = TN = FP = FN = 0
    num_macros = len(preds) // macro_size
    for i in range(num_macros):
        start = i * macro_size
        end = start + macro_size
        macro_preds = preds[start:end]
        macro_targets = targets[start:end]
        machine_alarmed = 1 if np.sum(macro_preds) >= 25 else 0
        machine_actually_broken = 1 if np.sum(macro_targets) > 0 else 0
        if machine_alarmed == 1 and machine_actually_broken == 1: TP += 1
        elif machine_alarmed == 0 and machine_actually_broken == 0: TN += 1
        elif machine_alarmed == 1 and machine_actually_broken == 0: FP += 1
        elif machine_alarmed == 0 and machine_actually_broken == 1: FN += 1
    acc = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0
    print(f"Revolution Accuracy  : {acc*100:.2f}%")
    print(f"Counts: TP={TP}, TN={TN}, FP={FP}, FN={FN}")

def evaluate_hardware(best_chromosome, X_test, Y_test_binary):
    W1 = torch.tensor(best_chromosome[0:Config.HIDDEN_FEATURES], dtype=torch.int32, device=device).unsqueeze(0)
    W2 = torch.tensor(best_chromosome[Config.HIDDEN_FEATURES:16], dtype=torch.int32, device=device).unsqueeze(0)
    T1 = torch.tensor([best_chromosome[16]], dtype=torch.int32, device=device)
    T2 = torch.tensor([best_chromosome[17]], dtype=torch.int32, device=device)
    X_test_batch = torch.tensor(X_test, dtype=torch.int32, device=device).unsqueeze(0)

    preds = simulate_hardware_population_pt(X_test_batch, W1, W2, T1, T2)[0].cpu().numpy()

    TP = np.sum((preds == 1) & (Y_test_binary == 1))
    TN = np.sum((preds == 0) & (Y_test_binary == 0))
    FP = np.sum((preds == 1) & (Y_test_binary == 0))
    FN = np.sum((preds == 0) & (Y_test_binary == 1))
    acc = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0

    print("\n" + "="*50)
    print("📊 MICRO-WINDOW EVALUATION")
    print("="*50)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Counts: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    evaluate_macro_window(preds, Y_test_binary, Config.MACRO_WINDOW_SIZE)

# =====================================================================
# 7. VHDL EXPORT & PYTHON FORMATTED WEIGHTS
# =====================================================================
def export_vhdl_and_vectors(best_chromosome, X_test, Y_test_binary, Y_test_multi):
    W1 = best_chromosome[0:Config.HIDDEN_FEATURES]
    W2 = best_chromosome[Config.HIDDEN_FEATURES:16]
    T1 = best_chromosome[16]
    T2 = best_chromosome[17]

    print("\n" + "="*65)
    print("🏆 WINNING GENOME (COPY & PASTE TO BENCHMARK SCRIPT)")
    print("="*65)
    print(f"W1_ARRAY = {list(W1)}")
    print(f"W2_ARRAY = {list(W2)}")
    print(f"T1_VAL = {T1}")
    print(f"T2_VAL = {T2}")
    print("="*65)

    fc1_str = ",\n        ".join([f"to_signed({w}, 8)" for w in W1])
    fc2_str = ",\n        ".join([f"to_signed({w}, 8)" for w in W2])
    vhdl_code = f"""library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

package snn_weights_pkg is
    constant INPUT_SIZE  : integer := 1;
    constant HIDDEN_SIZE : integer := {Config.HIDDEN_FEATURES};

    constant THRESHOLD_L1 : signed(31 downto 0) := to_signed({T1}, 32);
    constant THRESHOLD_L2 : signed(31 downto 0) := to_signed({T2}, 32);

    type weight_array is array (0 to HIDDEN_SIZE-1) of signed(7 downto 0);

    constant FC1_WEIGHTS : weight_array := (
        {fc1_str}
    );
    constant FC2_WEIGHTS : weight_array := (
        {fc2_str}
    );
end package snn_weights_pkg;
"""
    with open("snn_weights_pkg.vhd", "w") as f:
        f.write(vhdl_code)
    print("\n✅ VHDL Package created (snn_weights_pkg.vhd)")

    # Generate test vectors
    class_data = {0: [], 1: [], 2: [], 3: []}
    for i in range(len(X_test)):
        class_data[Y_test_multi[i]].append(X_test[i])

    # 🟢 EXACT BOUNDARY FIX: 525 windows = exactly 7 macro-windows per phase
    PHASE_WINDOWS = 525

    with open("adc_presentation_vectors.txt", "w") as f:
        def write_phase(label_idx, count):
            limit = min(count, len(class_data[label_idx]))
            for w in range(limit):
                for adc_val in class_data[label_idx][w]:
                    f.write(f"{adc_val}\n")
        write_phase(0, PHASE_WINDOWS)
        write_phase(1, PHASE_WINDOWS)
        write_phase(0, PHASE_WINDOWS)
        write_phase(2, PHASE_WINDOWS)
        write_phase(0, PHASE_WINDOWS)
        write_phase(3, PHASE_WINDOWS)
    print("✅ Test vectors saved to adc_presentation_vectors.txt")

    if IN_COLAB:
        files.download('snn_weights_pkg.vhd')
        files.download('adc_presentation_vectors.txt')

# =====================================================================
# 8. MAIN
# =====================================================================
if __name__ == "__main__":
    start_time = time.time()
    train_data, test_data = build_datasets()
    X_train, Y_train_bin, Y_train_multi, X_train_crucial = train_data
    X_test, Y_test_bin, Y_test_multi, X_test_crucial = test_data

    # Evolve
    best_genome = run_genetic_algorithm(X_train, Y_train_bin, Y_train_multi, X_train_crucial)

    # Evaluate on test set
    evaluate_hardware(best_genome, X_test, Y_test_bin)

    # Export for FPGA
    export_vhdl_and_vectors(best_genome, X_test, Y_test_bin, Y_test_multi)

    print(f"\n🚀 Total time: {int(time.time() - start_time)} seconds.")