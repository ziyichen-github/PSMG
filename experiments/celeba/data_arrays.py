import torch
import numpy as np

# Load the .pt file
file_path = 'all_results_celeba.pt'
data = torch.load(file_path)

# Initialize a dictionary to hold the averaged results
averaged_results = {}

# Extract unique methods, excluding 'single_task'
methods = {key[0] for key in data.keys() if key != 'single_task'}

# Calculate the average for each method
for method in methods:
    # Extract the arrays for the three repeats
    arrays = [data[(method, repeat)] for repeat in [10000, 20000, 30000] if (method, repeat) in data]
    
    # Compute the average array if all three repeats are available
    if len(arrays) == 3:
        averaged_array = np.mean(arrays, axis=0)
        averaged_results[method] = averaged_array

# Convert the dictionary to a readable format with 6 decimal places
averaged_results_formatted = {method: [round(float(value), 6) for value in array] for method, array in averaged_results.items()}

# Print the final results in the desired format
for method, values in averaged_results_formatted.items():
    print(f"'{method}': {values},")
