import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Define the range and number of elements
start, end = 58.8967, 53.5621
num_elements = 59

# Generate random values within the range
random_values = np.random.uniform(low=end, high=start, size=num_elements)

# Sort the array in decreasing order
sorted_values = np.sort(random_values)[::-1]
# Sort the array in increasing order
# sorted_values = np.sort(random_values)


# Generate a new array of random values within the range [-0.01, 0.01]
noise_values = np.random.uniform(low=-1, high=1, size=num_elements)

# Add the new array to the previously sorted values
adjusted_values = sorted_values + noise_values

# Adjust the precision to four significant digits after addition
final_values = list(np.round(adjusted_values, 4))

# Output the formatted sorted array
print(final_values)
