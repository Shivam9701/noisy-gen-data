import os
import csv
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from snr_calculator import calculate_snr

def spectral_gating(input_file, output_directory):
    # Load the audio file
    sample_rate, samples = wavfile.read(input_file)

    # Compute the spectrogram of the audio signal
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    # Compute the threshold for each frequency band
    threshold = np.mean(spectrogram, axis=1)

    # Apply the threshold to the spectrogram
    gated_spectrogram = np.where(spectrogram > threshold[:, np.newaxis], spectrogram, 0)

    # Reconstruct the audio signal from the gated spectrogram
    _, output_samples = signal.istft(gated_spectrogram, sample_rate)

    # Save the output audio file
    
    filtered_file_path = os.path.join(output_directory, 'filtered_' + os.path.basename(input_file))
    wavfile.write(filtered_file_path, sample_rate, np.int16(output_samples))
    return filtered_file_path



# Define input and output directories
# Get a list of all .wav files in the directory
input_directory = 'NoisySpeech_training'
output_directory = 'Spectral_Gating'
files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Dictionary to hold SNR values
snr_values = {}

# Apply the Kalman filter to each file and compare SNR
for file in files:
    original_file_path = os.path.join(input_directory, file)
    
    # Calculate original SNR from file name
    original_snr = float(file.split('_')[2])
    
    # Apply Kalman filter
    filtered_file_path = spectral_gating(original_file_path, output_directory)
    
    # Calculate new SNR
    new_snr = calculate_snr(filtered_file_path)
    
    # Store SNR values in dictionary
    snr_values[file] = {'original_snr': original_snr, 'new_snr': new_snr}

# Write SNR values to CSV file
with open('spectral_gating_snr_values.csv', 'w', newline='') as csvfile:
    fieldnames = ['file', 'original_snr', 'new_snr']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for file, snr in snr_values.items():
        writer.writerow({'file': file, 'original_snr': snr['original_snr'], 'new_snr': snr['new_snr']})