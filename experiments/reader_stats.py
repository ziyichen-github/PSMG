import torch

# Load the saved data
loaded_data = torch.load(
    f"experiments/nyuv2/save/pmgd_sd42_lr0.0001_bs2_101657.stats")
# print(loaded_data)
# Access the saved values
avg_cost = loaded_data["avg_cost"]
loss_list = loaded_data["losses"]
deltas = loaded_data["delta_m"]

# Use the loaded data as needed
print("Average Cost:", avg_cost)
# print("Loss List:", loss_list)
print("Deltas:", deltas)
