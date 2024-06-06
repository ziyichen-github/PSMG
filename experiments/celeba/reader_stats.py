import torch
# Replace 'path_to_file.pt' with the path to your .pt file
file_path = 'all_results_celeba.pt'
# Load the file
data = torch.load(file_path)
# Print the loaded data
print(data)

# Load the saved data
loaded_data = torch.load(
    f"/home/mx6835/Academic/MM1204/FAMO/experiments/celeba/save/pmgdn_sd42_050950.stats")
# print(loaded_data['metric'][4])
# loaded_data = torch.load(
#     f"/home/mx6835/Academic/MM1204/FAMO/experiments/celeba/save/ls_sd42.stats")
# print(loaded_data)
# Access the saved values
# avg_cost = loaded_data["avg_cost"]
# loss_list = loaded_data["losses"]
# deltas = loaded_data["delta_m"]

# Use the loaded data as needed
# print("Average Cost:", avg_cost)
# print("Loss List:", loss_list)
# print("Deltas:", deltas)
# [0.66314465 0.6998716  0.8265374  0.6106443  0.71039355 0.8564706
#  0.50622475 0.6126753  0.6380781  0.8371299  0.29906544 0.66084486
#  0.68001616 0.5364478  0.54196155 0.9318453  0.6644182  0.6471127
#  0.8841846  0.8619348  0.9660285  0.91927344 0.5061461  0.46894577
#  0.96648127 0.3970516  0.49762282 0.43706706 0.32224825 0.60596544
#  0.6901885  0.92159003 0.49457145 0.7307542  0.7002012  0.8736718
#  0.9419575  0.37763807 0.73659676 0.9206073 ]
