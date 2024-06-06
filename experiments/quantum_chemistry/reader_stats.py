import torch

# Load the saved data
loaded_data = torch.load(
    f"/home/mx6835/Academic/MM1204/FAMO/experiments/quantum_chemistry/save/save_remote/pmgdn_sd42_lr0.001_bs120_160629.stats")

# Access the saved values
avg_cost = loaded_data["avg_cost"]
loss_list = loaded_data["losses"]
deltas = loaded_data["delta_m"]
# 160629 299 
# Use the loaded data as needed
print("Average Cost:", avg_cost[299][2:2+11])
# print("Average Cost:", avg_cost[-1])
# print("Values: ", avg_cost[-1][2:2+11])
# print("Size of Average Cost:", avg_cost.shape)
# print("Loss List:", loss_list)
# print("Deltas:", deltas)
# print("Size of Deltas:", len(deltas))
