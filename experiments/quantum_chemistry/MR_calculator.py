import numpy as np
from scipy.stats import rankdata

# Data from the image
data = {
    'ls': [0.11, 0.33, 73.6, 89.7, 5.20, 14.06, 143.4, 144.2, 144.6, 140.3, 0.13],
    'si': [0.31, 0.35, 149.8, 135.7, 1.00, 4.51, 55.3, 55.8, 55.8, 55.3, 0.11],
    'rlw': [0.11, 0.34, 76.9, 92.8, 5.87, 15.47, 156.3, 157.1, 157.6, 153.0, 0.14],
    'dwa': [0.11, 0.33, 74.1, 90.6, 5.09, 13.99, 142.3, 143.0, 143.4, 139.3, 0.13],
    'uw': [0.39, 0.43, 166.2, 155.8, 1.07, 4.99, 66.4, 66.8, 66.8, 66.2, 0.12],
    'mgda': [0.22, 0.37, 126.8, 104.6, 3.23, 5.69, 88.4, 89.4, 89.3, 88.0, 0.12],
    'pcgrad': [0.11, 0.29, 75.9, 88.3, 3.94, 9.15, 116.4, 116.8, 117.2, 114.5, 0.11],
    'cagrad': [0.12, 0.32, 83.5, 94.8, 3.22, 6.93, 114.0, 114.3, 114.5, 112.3, 0.12],
    'imtlg': [0.14, 0.29, 98.3, 93.9, 1.75, 5.70, 101.4, 102.4, 102.0, 100.1, 0.10],
    'nashmtl': [0.10, 0.25, 82.9, 81.9, 2.43, 5.38, 74.5, 75.0, 75.1, 74.2, 0.09],
    'famo': [0.15, 0.30, 94.0, 95.2, 1.63, 4.95, 70.82, 71.2, 71.2, 70.3, 0.10],
    'pmgd': [0.12, 0.25, 77.22, 74.36, 3.01, 6.61, 102.99, 103.47, 103.69, 101.63, 0.09],
    'fairgrad': [0.12, 0.25, 87.57, 84.00, 2.15, 5.07, 70.89, 71.17, 71.21, 70.88, 0.10]
}


# Convert the data dictionary to a NumPy array
results = np.array([data[method] for method in data])

# Initialize the rank matrix
ranks = np.zeros_like(results)

# Calculate ranks per task, accounting for ties correctly
for j in range(results.shape[1]):
    ranks[:, j] = rankdata(results[:, j], method='min')

# Compute the mean rank for each method
method_rank = ranks.mean(axis=1)

# Display results
method_names = list(data.keys())
print("Ranks for each method in each task:")
for i, method in enumerate(method_names):
    rank_list = ranks[i, :]
    rank_str = ", ".join(f"{rank}" for rank in rank_list)
    print(f"{method}: [{rank_str}] - Mean Rank: {method_rank[i]:.2f}")
