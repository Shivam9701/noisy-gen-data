import os
import csv
import numpy as np
import scipy.io.wavfile as wavfile
from snr_calculator  import calculate_snr
from scipy.signal import lfilter, firwin

def equalization(input_file, output_directory):
    sample_rate, samples = wavfile.read(input_file)
    # Create a FIR filter
    taps = firwin(numtaps=30, cutoff=0.5)
    # Apply the filter to the samples
    output_samples = lfilter(taps, 1.0, samples)
    
    filtered_file_path = os.path.join(output_directory, 'filtered_' + os.path.basename(input_file))    
    wavfile.write(filtered_file_path, sample_rate, np.int16(output_samples))
    return filtered_file_path

input_directory = 'NoisySpeech_training'
output_directory = 'Equalization'
files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]
os.makedirs(output_directory, exist_ok=True)

snr_values = {}
for file in files:
    original_file_path = os.path.join(input_directory, file)
    original_snr = float(file.split('_')[2])
    filtered_file_path = equalization(original_file_path, output_directory)
    new_snr = calculate_snr(filtered_file_path)
    snr_values[file] = {'original_snr': original_snr, 'new_snr': new_snr}

with open('Equalization.csv', 'w', newline='') as csvfile:
    fieldnames = ['file', 'original_snr', 'new_snr']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for file, snr in snr_values.items():
        writer.writerow({'file': file, 'original_snr': snr['original_snr'], 'new_snr': snr['new_snr']})
filtered_file_path