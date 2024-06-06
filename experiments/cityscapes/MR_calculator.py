import numpy as np
from scipy.stats import rankdata

# Data from the image
data = {
    'LS': [75.18, 93.49, 0.0155, 46.77],
    'SI': [70.95, 91.73, 0.0161, 33.83],
    'RLW': [74.57, 93.41, 0.0158, 47.79],
    'DWA': [75.24, 93.52, 0.0160, 44.37],
    'UW': [72.02, 92.85, 0.0140, 30.13],
    'MGDA': [68.84, 91.54, 0.0309, 33.50],
    'PCGRAD': [75.13, 93.48, 0.0154, 42.07],
    'GRADDROP': [75.27, 93.53, 0.0157, 47.54],
    'CAGRAD': [75.16, 93.48, 0.0141, 37.60],
    'IMTL_G': [75.33, 93.49, 0.0135, 38.41],
    'MoCo': [75.42, 93.55, 0.0149, 34.19],
    'NASH_MTL': [75.41, 93.66, 0.0129, 35.02],
    'FAMO': [74.54, 93.29, 0.0145, 32.59],
    'FAIRGRAD': [75.72, 93.68, 0.0134, 32.25],
    'SDMGRAD': [74.53, 93.52, 0.0137, 34.01],
    'PMGD': [74.90, 93.37, 0.0135, 35.78],
}
# TEST: 0.2132 0.7490 0.9337 0.0135 | 0.0135 35.7773| 8.760

# Convert the data dictionary to a NumPy array
results = np.array([data[method] for method in data])

# Initialize the rank matrix
ranks = np.zeros_like(results)

# Calculate ranks per task, accounting for ties correctly
for j in range(results.shape[1]):
    ranks[:, j] = rankdata(results[:, j], method='min')

# Adjusting ranks for the specific tasks
reverse_tasks = [0, 1]  # Tasks that need the reverse ranking
for j in reverse_tasks:
    ranks[:, j] = 17 - ranks[:, j]

# Compute the mean rank for each method
method_rank = ranks.mean(axis=1)

# Display results
method_names = list(data.keys())
print("Ranks for each method in each task:")
for i, method in enumerate(method_names):
    rank_list = ranks[i, :]
    rank_str = ", ".join(f"{rank}" for rank in rank_list)
    print(f"{method}: [{rank_str}] - Mean Rank: {method_rank[i]:.2f}")
