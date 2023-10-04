# snr_calculator.py

import numpy as np
import scipy.io.wavfile as wav

def calculate_snr(file_path):
    # Read file
    sample_rate, data = wav.read(file_path)
    
    # Calculate Signal Power
    signal_power = np.mean(data ** 2)
    
    # Generate White Gaussian noise
    noise = np.random.normal(0, 1, len(data))
    
    # Calculate Noise Power
    noise_power = np.mean(noise ** 2)
    
    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr
