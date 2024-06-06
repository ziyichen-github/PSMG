# import numpy as np
# from scipy.stats import rankdata

# # Data from the image
# data = {
#     'ls': [39.29, 65.33, 0.5493, 0.2263, 28.15, 23.96, 20.29, 47.50, 61.08],
#     'si': [38.45, 64.27, 0.5354, 0.2201, 27.60, 23.37, 22.53, 48.87, 62.32],
#     'rl': [37.17, 63.77, 0.5719, 0.2405, 27.67, 23.18, 22.16, 47.05, 61.22],
#     'dlwa': [39.11, 65.31, 0.5550, 0.2281, 27.21, 24.14, 22.47, 50.78, 62.30],
#     'uw': [36.87, 63.17, 0.5446, 0.2260, 27.04, 22.61, 23.54, 49.05, 61.35],
#     'mgda': [30.47, 59.90, 0.6070, 0.2555, 24.88, 19.45, 29.18, 56.83, 65.55],
#     'pcgrad': [38.06, 64.12, 0.5550, 0.2235, 27.41, 22.80, 22.35, 48.98, 62.64],
#     'graddrop': [39.39, 65.12, 0.5485, 0.2279, 27.48, 22.96, 23.52, 49.44, 63.14],
#     'cagrad': [39.79, 65.49, 0.5456, 0.2250, 27.31, 21.58, 23.84, 49.63, 62.80],
#     'imtl_g': [39.35, 64.72, 0.5426, 0.2256, 26.92, 21.19, 26.13, 52.23, 63.70],
#     'nashmtl': [40.13, 65.93, 0.5261, 0.2171, 25.26, 20.08, 28.40, 50.77, 64.69],
#     'famo': [38.88, 64.90, 0.5474, 0.2194, 25.06, 19.57, 29.97, 56.61, 64.98],
#     'pmgd': [38.39, 65.04, 0.5536, 0.2214, 25.81, 20.32, 28.41, 55.02, 67.30]
# }


# # Convert the data dictionary to a NumPy array
# results = np.array([data[method] for method in data])

# # Initialize the rank matrix
# ranks = np.zeros_like(results)

# # Calculate ranks per task, accounting for ties correctly
# for j in range(results.shape[1]):
#     ranks[:, j] = rankdata(results[:, j], method='min')

# # Compute the mean rank for each method
# method_rank = ranks.mean(axis=1)

# # Display results
# method_names = list(data.keys())
# print("Ranks for each method in each task:")
# for i, method in enumerate(method_names):
#     rank_list = ranks[i, :]
#     rank_str = ", ".join(f"{rank}" for rank in rank_list)
#     print(f"{method}: [{rank_str}] - Mean Rank: {method_rank[i]:.2f}")
import numpy as np
from scipy.stats import rankdata

# Data from the image
data = {
    'ls': [39.29, 65.33, 0.5493, 0.2263, 28.15, 23.96, 22.09, 47.50, 61.08],
    'si': [38.45, 64.27, 0.5354, 0.2201, 27.60, 23.37, 22.53, 48.57, 62.32],
    'rlw': [37.17, 63.77, 0.5759, 0.2410, 28.27, 24.18, 22.26, 47.05, 60.62],
    'dwa': [39.11, 65.31, 0.5510, 0.2285, 27.61, 23.18, 24.17, 50.18, 62.29],
    'uw': [36.87, 63.17, 0.5446, 0.2260, 27.04, 22.61, 23.54, 49.05, 63.65],
    'mgda': [30.47, 59.90, 0.6070, 0.2555, 24.88, 19.45, 29.18, 56.88, 69.36],
    'pcgrad': [38.06, 64.64, 0.5550, 0.2325, 27.41, 22.80, 23.86, 49.83, 63.14],
    'graddrop': [39.39, 65.12, 0.5455, 0.2279, 27.48, 22.96, 23.38, 49.44, 62.87],
    'cagrad': [39.79, 65.49, 0.5486, 0.2250, 26.31, 21.58, 25.61, 52.36, 65.58],
    'imtl_g': [39.35, 65.60, 0.5426, 0.2256, 26.02, 21.19, 26.20, 53.13, 66.24],
    'nashmtl': [40.13, 65.93, 0.5261, 0.2171, 25.26, 20.58, 28.40, 55.47, 68.15],
    'famo': [38.88, 64.90, 0.5474, 0.2194, 25.06, 19.57, 29.21, 56.61, 68.98],
    'fairgrad': [39.74, 66.01, 0.5377, 0.2236, 24.84, 19.60, 29.26, 56.58, 69.16],
    'sdmgrad': [40.47, 65.90, 0.5225, 0.2084, 25.07, 19.99, 28.54, 55.74, 68.53],
    'pmgd': [35.44, 63.78, 0.5496, 0.2369, 24.83, 18.89, 30.68, 58.00, 69.84]
}

# Convert the data dictionary to a NumPy array
results = np.array([data[method] for method in data])

# Initialize the rank matrix
ranks = np.zeros_like(results)

# Calculate ranks per task, accounting for ties correctly
for j in range(results.shape[1]):
    ranks[:, j] = rankdata(results[:, j], method='min')

# Adjusting ranks for the specific tasks
reverse_tasks = [0, 1, 6, 7, 8]  # Tasks that need the reverse ranking
for j in reverse_tasks:
    ranks[:, j] = 16 - ranks[:, j]

# Compute the mean rank for each method
method_rank = ranks.mean(axis=1)

# Display results
method_names = list(data.keys())
print("Ranks for each method in each task:")
for i, method in enumerate(method_names):
    rank_list = ranks[i, :]
    rank_str = ", ".join(f"{rank}" for rank in rank_list)
    print(f"{method}: [{rank_str}] - Mean Rank: {method_rank[i]:.2f}")
