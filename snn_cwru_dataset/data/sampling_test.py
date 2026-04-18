import scipy.io as sio
import numpy as np

def check_sampling_rate(filepath):
    mat = sio.loadmat(filepath)
    
    # Method 1: If there is a 'time' variable
    if 'time' in mat:
        t = mat['time'].flatten()
        duration = t[-1] - t[0]
        num_samples = len(t)
        fs = num_samples / duration
        print(f"{filepath}: time array length {num_samples}, duration {duration:.2f}s -> fs = {fs:.0f} Hz")
        return fs
    
    # Method 2: Look for DE_time or FE_time
    for key in ['DE_time', 'FE_time']:
        if key in mat:
            data = mat[key].flatten()
            # The dataset often uses a fixed duration of 10 seconds for 12k and 48k?
            # Alternatively, if RPM is known, you can estimate, but simpler: 
            # The original CWRU data for 12k has 121,265 samples (~10.1s) for 12k? Actually typical 12k files have 121,265 samples.
            # But better: just compute based on expected rates.
            # We can try to infer by looking at length: 12k for 10s = 120,000 samples; 48k for 10s = 480,000 samples.
            num_samples = len(data)
            if num_samples > 200000:
                fs = 48000
            elif num_samples > 100000:
                fs = 12000
            else:
                fs = -1
            print(f"{filepath}: {key} length {num_samples} -> inferred fs = {fs} Hz")
            return fs
    print(f"{filepath}: Could not determine sampling rate.")
    return None

# Example usage:
check_sampling_rate('97.mat')
# check_sampling_rate('98.mat')