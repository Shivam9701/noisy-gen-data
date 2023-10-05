import os
import csv
import numpy as np
import scipy.io.wavfile as wav
import pywt
from snr_calculator import calculate_snr  

def wavelet_denoise(data, wavelet, level):
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(data, wavelet, mode="per")
    
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    
    # Calculate the univeral threshold
    uthresh = sigma * np.sqrt(2*np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')

def apply_wavelet_denoise(file_path, output_directory):
    # Read file
    sample_rate, data = wav.read(file_path)
    
    # Apply wavelet denoise
    filtered_data = wavelet_denoise(data, 'db8', 2)
    
    # Write filtered data to new .wav file in the specified output directory
    filtered_file_path = os.path.join(output_directory, 'filtered_' + os.path.basename(file_path))
    wav.write(filtered_file_path, sample_rate, np.int16(filtered_data))
    
    return filtered_file_path

# Get a list of all .wav files in the directory
input_directory = 'NoisySpeech_training'
output_directory = 'Wavelet_Denoised'
files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Dictionary to hold SNR values
snr_values = {}

# Apply the wavelet denoise to each file and compare SNR
for file in files:
    original_file_path = os.path.join(input_directory, file)
    
    # Calculate original SNR from file name
    original_snr = float(file.split('_')[2])
    
    # Apply wavelet denoise
    filtered_file_path = apply_wavelet_denoise(original_file_path, output_directory)
    
    # Calculate new SNR
    new_snr = calculate_snr(filtered_file_path)
    
    # Store SNR values in dictionary
    snr_values[file] = {'original_snr': original_snr, 'new_snr': new_snr}

# Write SNR values to CSV file
with open('wavelet_snr_values.csv', 'w', newline='') as csvfile:
    fieldnames = ['file', 'original_snr', 'new_snr']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for file, snr in snr_values.items():
        writer.writerow({'file': file, 'original_snr': snr['original_snr'], 'new_snr': snr['new_snr']})
