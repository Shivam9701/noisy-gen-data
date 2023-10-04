import os
import csv
import numpy as np
import scipy.io.wavfile as wav
from snr_calculator import calculate_snr  

def kalman_filter(data, Q, R):
    # Initialize state and covariance
    x = np.zeros(len(data))
    P = np.zeros(len(data))
    
    # Apply Kalman filter
    for i in range(1, len(data)):
        # Prediction step
        x[i] = x[i-1]
        P[i] = P[i-1] + Q

        # Update step
        K = P[i] / (P[i] + R) # Kalman gain
        x[i] = x[i] + K * (data[i] - x[i])
        P[i] = (1 - K) * P[i]
    
    return x

def apply_kalman_filter(file_path, output_directory):
    # Read file
    sample_rate, data = wav.read(file_path)
    
    # Apply Kalman filter
    filtered_data = kalman_filter(data, Q=1e-5, R=0.01**2)
    
    # Write filtered data to new .wav file in the specified output directory
    filtered_file_path = os.path.join(output_directory, 'filtered_' + os.path.basename(file_path))
    wav.write(filtered_file_path, sample_rate, np.int16(filtered_data))
    
    return filtered_file_path

# Get a list of all .wav files in the directory
input_directory = 'NoisySpeech_training'
output_directory = 'Kalman_Filtered'
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
    filtered_file_path = apply_kalman_filter(original_file_path, output_directory)
    
    # Calculate new SNR
    new_snr = calculate_snr(filtered_file_path)
    
    # Store SNR values in dictionary
    snr_values[file] = {'original_snr': original_snr, 'new_snr': new_snr}

# Write SNR values to CSV file
with open('kalman_snr_values.csv', 'w', newline='') as csvfile:
    fieldnames = ['file', 'original_snr', 'new_snr']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for file, snr in snr_values.items():
        writer.writerow({'file': file, 'original_snr': snr['original_snr'], 'new_snr': snr['new_snr']})
