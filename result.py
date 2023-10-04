import pandas as pd
import os

# List of files to process
files = ['Equalization.csv', 'kalman_snr_values.csv', 'spectral_gating_snr_values.csv', 'wiener_snr_values.csv']

for file in files:
    # Read the CSV file
    df = pd.read_csv(file)

    # Group by 'original_snr' and calculate the average of 'new_snr'
    grouped = df.groupby('original_snr')['new_snr'].mean()

    # Convert the grouped data back to a DataFrame and reset index
    avg_df = pd.DataFrame(grouped).reset_index()

    # Rename columns
    avg_df.columns = ['Original_snr', 'avg_new_snr']

    # Add a new column for the noise reduction method
    avg_df['Method'] = file.split('.')[0]  # Use the filename (without extension) as the method name

    # Reorder columns
    avg_df = avg_df[['Method', 'Original_snr', 'avg_new_snr']]

    # Check if the file exists
    if os.path.isfile('Filtered_Averages.csv'):
        # If it exists, append without writing the header
        avg_df.to_csv('Filtered_Averages.csv', mode='a', header=False, index=False)
    else:
        # If it doesn't exist, write the DataFrame to a new CSV file with a header
        avg_df.to_csv('Filtered_Averages.csv', index=False)
