# import numpy as np
# values = [100.137, 100.137, 97.269, 97.269, 103.245, 103.245, 104.582, 104.582, 103.738, 103.738, 102.569, 102.569, 102.438,
#           102.438, 108.238, 108.238, 105.470, 105.470, 104.39, 104.39, 101.86, 101.86, 95.454, 95.454, 95.432, 95.432, 95.0]
# arr = np.array(values)
# mean = np.mean(arr)
# std_err = np.std(arr) / np.sqrt(len(arr))
# print(f"Mean: {mean:.3f}")
# print(f"Standard Error: {std_err:.3f}")
# Input data
data = """
ls_7875: 1.72 44 99 300
mgda_9779: 13.24 33 57 300
pcgrad_7876: 6.18 42 102 300
cagrad_7877: 5.31 44 85 300
imtlg_6825: 5.40 51 112 259
famo_6733: 1.96 37 67 224
nashmtl_8608: 8.45 34 44 103
fairgrad_5011: 5.42 30 58 299
pmgd_6734: 2.23 31 49 293
"""

# Process each line of the data
lines = data.strip().split('\n')
results = []

for line in lines:
    parts = line.split(': ')
    method = parts[0]
    values = list(map(float, parts[1].split()))
    first_value = values[0]
    # results_line = f"{method}: {first_value * values[1]:.2f} {first_value * values[2]:.2f} {first_value * values[3]:.2f}"
    results_line = f"{first_value * values[1]:.2f} {first_value * values[2]:.2f} {first_value * values[3]:.2f}"
    results.append(results_line)

# Print the results
for result in results:
    print(result)
