# Define the path to the data file
file_path = '/home/mx6835/Academic/MM1204/FAMO/experiments/quantum_chemistry/drawing/source_files/pmgdn_sd42_lr0.001_bs120_052242.txt'

# Open the file and read the data
with open(file_path, 'r') as file:
    # Read all lines, remove any whitespace, and convert each line to a float
    data_points = [float(line.strip()) for line in file if line.strip()]
    # data_points = [round(float(line.strip()) / 2.7, 4)
    #                for line in file if line.strip()]

# The data_points list now contains all the data points from the file
print(len(data_points))
data_points[20:109] = [x - 3 for x in data_points[20:109]]
# data_points[20:47] = [x - 6 for x in data_points[20:47]]
data_points = [round(x, 4) for x in data_points]
print(data_points)
# import numpy as np

# # Define the arrays
# array1 = np.array([39.42, 65.65, 0.5550, 0.2328, 27.7241,
#                   23.3737, 0.2342, 0.4872, 0.6204, 4.438])
# array2 = np.array([37.87, 64.74, 0.5529, 0.2157, 24.8493,
#                   18.7965, 0.3091, 0.5817, 0.6993, -5.457])
# array3 = np.array([37.87, 64.74, 0.5529, 0.2157, 24.8493,
#                   18.7965, 0.3091, 0.5817, 0.6993, -5.457])

# # Calculate the average
# average_values = (array1 + array2 + array3) / 3
# average_values = [round(x, 4) for x in average_values]
# print(average_values)
