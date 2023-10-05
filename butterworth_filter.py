import os
import csv
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from snr_calculator import calculate_snr

def butter_lowpass_filter(input_file, output_directory, cutoff, order=5):
    # Load the audio file
    sample_rate, samples = wavfile.read(input_file)

    # Normalize frequency for the Butterworth filter
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist

    # Design and apply the Butterworth filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    output_samples = signal.lfilter(b, a, samples)

    # Save the output audio file
    filtered_file_path = os.path.join(output_directory, 'filtered_' + os.path.basename(input_file))
    wavfile.write(filtered_file_path, sample_rate, np.int16(output_samples))
    
    return filtered_file_path

# Define input and output directories
input_directory = 'NoisySpeech_training'
output_directory = 'Butterworth_Lowpass'
files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Dictionary to hold SNR values
snr_values = {}

for file in files:
    original_file_path = os.path.join(input_directory, file)
    
    # Calculate original SNR from file name
    original_snr = float(file.split('_')[2])

    filtered_file_path = butter_lowpass_filter(original_file_path, output_directory, cutoff=3000)
    
    # Calculate new SNR
    new_snr = calculate_snr(filtered_file_path)
    
    # Store SNR values in dictionary
    snr_values[file] = {'original_snr': original_snr, 'new_snr': new_snr}

# Write SNR values to CSV file
with open('butterworth_lowpass.csv', 'w', newline='') as csvfile:
    fieldnames = ['file', 'original_snr', 'new_snr']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for file, snr in snr_values.items():
        writer.writerow({'file': file, 'original_snr': snr['original_snr'], 'new_snr': snr['new_snr']})