# =====================================================================
# MULTI-DOMAIN NEUROEVOLUTIONARY SNN (1500 RPM + 900 RPM)
# Evolving 32 Heterogeneous Neurons across multiple operating speeds.
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Compute Device: {device}")

class Config:
    DATA_DIR = "/content/drive/MyDrive/data"

    # 🟢 NEW: 6-File Multi-Domain Training List
    TRAIN_FILES = [
        {"file": "N15_M07_F10_K001_1.mat", "class_id": 0},
        {"file": "N15_M07_F10_KI01_1.mat", "class_id": 1},
        {"file": "N15_M07_F10_KA01_1.mat", "class_id": 2},
        {"file": "N09_M07_F10_K001_1.mat", "class_id": 0},
        {"file": "N09_M07_F10_KI01_1.mat", "class_id": 1},
        {"file": "N09_M07_F10_KA01_1.mat", "class_id": 2}
    ]

    WINDOW_SIZE = 2048
    HIDDEN_FEATURES = 32
    ADC_VREF = 5.0
    MACRO_WINDOW_SIZE = 12

    POP_SIZE = 1000
    GENERATIONS = 1000
    MUTATION_RATE = 0.10       # 🟢 Increased base mutation for harder puzzle

    RAW_ONLY_GENERATIONS = 50
    BONUS_RAMP_START = 50
    BONUS_RAMP_END = 250
    MAX_BONUS_WEIGHT = 0.3
    PERF_THRESHOLD = 0.7
    ADAPTIVE_UPDATE_INTERVAL = 10

    FAULT_BANDS = {
        1: (0.125, 0.188),
        2: (0.039, 0.078),
    }

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

def quantize_to_8bit_adc(raw_signal):
    scaled = (np.abs(raw_signal) / Config.ADC_VREF) * 255.0
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
    print("📊 Building Multi-Domain Datasets (1500 RPM + 900 RPM)...")

    X_train_list, Y_bin_list, Y_multi_list = [], [], []
    X_test_list, Y_test_multi_list = [], []

    for item in Config.TRAIN_FILES:
        filename = item["file"]

        # 🟢 THE FIX: Safely extract the exact folder name (K001, KI01, KA01) from the filename string
        # N15_M07_F10_K001_1.mat -> splits into -> ['N15', 'M07', 'F10', 'K001', '1.mat']
        folder_name = filename.split('_')[3]

        filepath = os.path.join(Config.DATA_DIR, folder_name, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ ERROR: Cannot find {filepath}. Check if your Drive is mounted and the path is correct.")

        raw = extract_vibration_data(filepath)
        wins = create_windows(quantize_to_8bit_adc(raw), Config.WINDOW_SIZE)

        # 80/20 Train/Test Split per file
        split_idx = int(len(wins) * 0.80)

        X_train_list.append(wins[:split_idx])
        bin_label = 0 if item["class_id"] == 0 else 1
        Y_bin_list.append(np.full(split_idx, bin_label, dtype=np.int32))
        Y_multi_list.append(np.full(split_idx, item["class_id"], dtype=np.int32))

        X_test_list.append(wins[split_idx:])
        Y_test_multi_list.append(np.full(len(wins)-split_idx, item["class_id"], dtype=np.int32))

    X_train = np.concatenate(X_train_list, axis=0)
    Y_train_binary = np.concatenate(Y_bin_list, axis=0)
    Y_train_multi = np.concatenate(Y_multi_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    Y_test_multi = np.concatenate(Y_test_multi_list, axis=0)
    Y_test_binary = (Y_test_multi > 0).astype(np.int32)

    # 🟢 DYNAMIC SUBSAMPLING: Handle the larger 6-file dataset
    sample_size = min(4000, len(X_train))
    idx = np.random.choice(len(X_train), size=sample_size, replace=False)
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

def simulate_hardware_population_pt(X_raw, W1, W2, LEAKS, T1, T2):
    POP, SAMPLES, _ = X_raw.shape
    HIDDEN = Config.HIDDEN_FEATURES

    mem1 = torch.zeros((POP, SAMPLES, HIDDEN), dtype=torch.int32, device=device)
    mem2 = torch.zeros((POP, SAMPLES), dtype=torch.int32, device=device)
    micro_alarms = torch.zeros((POP, SAMPLES), dtype=torch.int32, device=device)

    w1_exp = W1.unsqueeze(1)
    w2_exp = W2.unsqueeze(1)
    leaks_exp = LEAKS.unsqueeze(1)
    t1_exp = T1.unsqueeze(1).unsqueeze(2)
    t2_exp = T2.unsqueeze(1)

    for step in range(Config.WINDOW_SIZE):
        x_t_exp = X_raw[:, :, step].unsqueeze(2)
        cur1 = x_t_exp * w1_exp
        mem1 = (mem1 >> leaks_exp) + cur1
        spk1 = (mem1 > t1_exp).to(torch.int32)
        mem1 = mem1 * (1 - spk1)

        cur2 = torch.sum(spk1 * w2_exp, dim=2)
        mem2 = (mem2 >> 1) + cur2
        spk2 = (mem2 > t2_exp).to(torch.int32)
        mem2 = mem2 * (1 - spk2)

        micro_alarms |= spk2

    return micro_alarms

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

def run_genetic_algorithm(X_train_raw, Y_train_binary, Y_train_multi, X_train_crucial):
    print(f"\n🧬 Evolving Multi-Domain SNN on {device}")

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

    HF = Config.HIDDEN_FEATURES
    GENE_SIZE = (HF * 3) + 2

    population = np.zeros((Config.POP_SIZE, GENE_SIZE), dtype=np.int32)
    population[:, 0:HF*2] = np.random.randint(-127, 127, size=(Config.POP_SIZE, HF*2))
    population[:, HF*2:HF*3] = np.random.randint(0, 4, size=(Config.POP_SIZE, HF))
    population[:, HF*3] = np.random.randint(1000, 20000, size=Config.POP_SIZE)
    population[:, HF*3+1] = np.random.randint(10, 1000, size=Config.POP_SIZE)

    best_chromosome = None
    best_fitness = 0.0

    def tournament_selection(pop, fitness_array, k=7): # 🟢 Increased to K=7 for harsher competition
        idx = np.random.randint(0, len(pop), size=k)
        best_idx = idx[np.argmax(fitness_array[idx])]
        return pop[best_idx]

    current_mutation_rate = Config.MUTATION_RATE
    stagnation_counter = 0

    for gen in range(Config.GENERATIONS):
        if gen < Config.RAW_ONLY_GENERATIONS:
            bonus_weight = 0.0
        else:
            ramp_progress = (gen - Config.RAW_ONLY_GENERATIONS) / (Config.BONUS_RAMP_END - Config.RAW_ONLY_GENERATIONS)
            bonus_weight = Config.MAX_BONUS_WEIGHT * np.clip(ramp_progress, 0.0, 1.0)

        W1 = torch.tensor(population[:, 0:HF], dtype=torch.int32, device=device)
        W2 = torch.tensor(population[:, HF:HF*2], dtype=torch.int32, device=device)
        LEAKS = torch.tensor(population[:, HF*2:HF*3], dtype=torch.int32, device=device)
        T1 = torch.tensor(population[:, HF*3], dtype=torch.int32, device=device)
        T2 = torch.tensor(population[:, HF*3+1], dtype=torch.int32, device=device)

        preds = simulate_hardware_population_pt(X_train_batch, W1, W2, LEAKS, T1, T2)
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
            stagnation_counter = 0
            current_mutation_rate = max(0.02, current_mutation_rate * 0.98)
        else:
            stagnation_counter += 1

        if stagnation_counter > 75:
            current_mutation_rate = 0.20  # 🟢 Harsher shock for multi-domain plateau
            stagnation_counter = 0
            print(f"   ⚡ STAGNATION SHOCK: Mutation rate spiked to 20%!")

        if gen >= Config.RAW_ONLY_GENERATIONS and (gen % Config.ADAPTIVE_UPDATE_INTERVAL == 0):
            W1_best = torch.tensor(best_chromosome[0:HF], dtype=torch.int32, device=device).unsqueeze(0)
            W2_best = torch.tensor(best_chromosome[HF:HF*2], dtype=torch.int32, device=device).unsqueeze(0)
            LEAKS_best = torch.tensor(best_chromosome[HF*2:HF*3], dtype=torch.int32, device=device).unsqueeze(0)
            T1_best = torch.tensor([best_chromosome[HF*3]], dtype=torch.int32, device=device)
            T2_best = torch.tensor([best_chromosome[HF*3+1]], dtype=torch.int32, device=device)

            val_preds = simulate_hardware_population_pt(X_val_batch, W1_best, W2_best, LEAKS_best, T1_best, T2_best)[0]

            recalls = {}
            new_weights = np.ones(n_bands)
            for i, cid in enumerate([1, 2]):
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
            if gen % 50 == 0:
                 print(f"   Gen {gen}: Updated Class Weights = {class_weights_t.cpu().numpy().round(3)}")

        if gen % 5 == 0:
            print(f"Gen {gen:03d} | Best Fit: {total_fitness[0]:.4f} (base={base_fitness[0]:.3f}, bonus={bonus[0]:.3f}) | T1={population[0,HF*3]}, T2={population[0,HF*3+1]}")

        next_gen = np.zeros_like(population)
        elite_count = int(Config.POP_SIZE * 0.1)
        next_gen[:elite_count] = population[:elite_count]

        mut_var_w = max(5, int(127 * (1.0 - (gen / Config.GENERATIONS))))
        mut_var_t1 = max(500, int(5000 * (1.0 - (gen / Config.GENERATIONS))))
        mut_var_t2 = max(20, int(200 * (1.0 - (gen / Config.GENERATIONS))))

        for i in range(elite_count, int(Config.POP_SIZE * 0.8)):
            p1 = tournament_selection(population, total_fitness, k=7)
            p2 = tournament_selection(population, total_fitness, k=7)

            mask = np.random.randint(0, 2, size=GENE_SIZE).astype(bool)
            child = np.where(mask, p1, p2)

            if np.random.rand() < current_mutation_rate:
                child[0:HF*2] += np.random.randint(-mut_var_w, mut_var_w, size=HF*2)
                child[HF*2:HF*3] += np.random.randint(-1, 2, size=HF)
                child[HF*3] += np.random.randint(-mut_var_t1, mut_var_t1)
                child[HF*3+1] += np.random.randint(-mut_var_t2, mut_var_t2)

            child[0:HF*2] = np.clip(child[0:HF*2], -127, 127)
            child[HF*2:HF*3] = np.clip(child[HF*2:HF*3], 0, 4)
            child[HF*3] = np.clip(child[HF*3], 500, 32000)
            child[HF*3+1] = np.clip(child[HF*3+1], 5, 2000)
            next_gen[i] = child

        random_start = int(Config.POP_SIZE * 0.8)
        next_gen[random_start:, 0:HF*2] = np.random.randint(-127, 127, size=(Config.POP_SIZE - random_start, HF*2))
        next_gen[random_start:, HF*2:HF*3] = np.random.randint(0, 4, size=(Config.POP_SIZE - random_start, HF))
        next_gen[random_start:, HF*3] = np.random.randint(1000, 20000, size=Config.POP_SIZE - random_start)
        next_gen[random_start:, HF*3+1] = np.random.randint(10, 1000, size=Config.POP_SIZE - random_start)

        population = next_gen

    print(f"✅ Evolution Complete. Best Fitness: {best_fitness:.4f}")
    return best_chromosome

def evaluate_macro_window(preds, targets, macro_size):
    print("\n" + "="*50)
    print(f"🏆 FINAL EVALUATION (MACRO-WINDOW / ~192ms)")
    print("="*50)
    TP = TN = FP = FN = 0
    num_macros = len(preds) // macro_size
    alarm_threshold = max(1, macro_size // 2)

    for i in range(num_macros):
        start = i * macro_size
        end = start + macro_size
        macro_preds = preds[start:end]
        macro_targets = targets[start:end]

        machine_alarmed = 1 if np.sum(macro_preds) >= alarm_threshold else 0
        machine_actually_broken = 1 if np.sum(macro_targets) > 0 else 0

        if machine_alarmed == 1 and machine_actually_broken == 1: TP += 1
        elif machine_alarmed == 0 and machine_actually_broken == 0: TN += 1
        elif machine_alarmed == 1 and machine_actually_broken == 0: FP += 1
        elif machine_alarmed == 0 and machine_actually_broken == 1: FN += 1

    acc = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0
    print(f"Revolution Accuracy  : {acc*100:.2f}%")
    print(f"Counts: TP={TP}, TN={TN}, FP={FP}, FN={FN}")

def evaluate_hardware(best_chromosome, X_test, Y_test_binary):
    HF = Config.HIDDEN_FEATURES
    W1 = torch.tensor(best_chromosome[0:HF], dtype=torch.int32, device=device).unsqueeze(0)
    W2 = torch.tensor(best_chromosome[HF:HF*2], dtype=torch.int32, device=device).unsqueeze(0)
    LEAKS = torch.tensor(best_chromosome[HF*2:HF*3], dtype=torch.int32, device=device).unsqueeze(0)
    T1 = torch.tensor([best_chromosome[HF*3]], dtype=torch.int32, device=device)
    T2 = torch.tensor([best_chromosome[HF*3+1]], dtype=torch.int32, device=device)
    X_test_batch = torch.tensor(X_test, dtype=torch.int32, device=device).unsqueeze(0)

    preds = simulate_hardware_population_pt(X_test_batch, W1, W2, LEAKS, T1, T2)[0].cpu().numpy()

    TP = np.sum((preds == 1) & (Y_test_binary == 1))
    TN = np.sum((preds == 0) & (Y_test_binary == 0))
    FP = np.sum((preds == 1) & (Y_test_binary == 0))
    FN = np.sum((preds == 0) & (Y_test_binary == 1))
    acc = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0

    print("\n" + "="*50)
    print("📊 MICRO-WINDOW EVALUATION (Multi-Domain Test Set)")
    print("="*50)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Counts: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    evaluate_macro_window(preds, Y_test_binary, Config.MACRO_WINDOW_SIZE)

def export_vhdl_and_vectors(best_chromosome, X_test, Y_test_binary, Y_test_multi):
    HF = Config.HIDDEN_FEATURES
    W1 = best_chromosome[0:HF]
    W2 = best_chromosome[HF:HF*2]
    LEAKS = best_chromosome[HF*2:HF*3]
    T1 = best_chromosome[HF*3]
    T2 = best_chromosome[HF*3+1]

    print("\n" + "="*65)
    print("🏆 WINNING GENOME (MULTI-DOMAIN - READY FOR MASS TEST)")
    print("="*65)
    print(f"W1_ARRAY = {list(W1)}")
    print(f"W2_ARRAY = {list(W2)}")
    print(f"LEAK_ARRAY = {list(LEAKS)}")
    print(f"T1_VAL = {T1}")
    print(f"T2_VAL = {T2}")
    print("="*65)

if __name__ == "__main__":
    start_time = time.time()
    train_data, test_data = build_datasets()
    X_train, Y_train_bin, Y_train_multi, X_train_crucial = train_data
    X_test, Y_test_bin, Y_test_multi, X_test_crucial = test_data

    best_genome = run_genetic_algorithm(X_train, Y_train_bin, Y_train_multi, X_train_crucial)
    evaluate_hardware(best_genome, X_test, Y_test_bin)
    export_vhdl_and_vectors(best_genome, X_test, Y_test_bin, Y_test_multi)

    print(f"\n🚀 Total time: {int(time.time() - start_time)} seconds.")