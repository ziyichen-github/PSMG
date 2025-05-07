from methods.weight_methods import WeightMethods
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from experiments.nyuv2.utils import ConfMatrix, delta_fn, depth_error, normal_error
from experiments.nyuv2.models import SegNet, SegNetMtan
from experiments.nyuv2.data import NYUv2
import logging
import os
import wandb
from argparse import ArgumentParser
import numpy as np
import torch
import time
import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import trange
import networkx as nx
_ = torch.cuda.empty_cache()

set_logger()


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) !=
                   0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss

def main(path, lr, bs, device):

    num_agents = args.num_agents
    network_sparsity = args.network_sparsity

    # Parameter validation
    if not 0 <= network_sparsity <= 1:
        raise ValueError("Network sparsity must be between 0 and 1")
    if num_agents < 1:
        raise ValueError("Number of agents must be at least 1")

    # Generate communication matrix W
    G = nx.erdos_renyi_graph(
        num_agents,
        p=(1 - network_sparsity),  # Convert sparsity to connection probability
        seed=args.seed
    )
    L = nx.laplacian_matrix(G).todense()
    max_eig = max(np.linalg.eigvals(L))
    W = np.identity(num_agents) - (2 / (3 * max_eig)) * L
    W_tensor = torch.tensor(W, dtype=torch.float32).to(device)
    
    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    # Split datasets into subsets for each agent
    nyuv2_train_set = NYUv2(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    train_size = len(nyuv2_train_set)
    train_split_sizes = [train_size // num_agents] * num_agents
    train_split_sizes[-1] += train_size % num_agents
    train_subsets = random_split(
        nyuv2_train_set,
        train_split_sizes,
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loaders = [
        DataLoader(
            subset,
            batch_size=bs,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + i)
        )
        for i, subset in enumerate(train_subsets)
    ]

    nyuv2_test_set = NYUv2(root=path.as_posix(), train=False)
    test_size = len(nyuv2_test_set)
    test_split_sizes = [test_size // num_agents] * num_agents
    test_split_sizes[-1] += test_size % num_agents
    test_subsets = random_split(
        nyuv2_test_set,
        test_split_sizes,
        generator=torch.Generator().manual_seed(args.seed)
    )

    test_loaders = [
        DataLoader(subset, batch_size=bs, shuffle=False)
        for subset in test_subsets
    ]

    # Initialize models and optimizers for each agent
    models = [SegNetMtan().to(device) for _ in range(num_agents)]
    
    weight_methods_parameters = extract_weight_method_parameters_from_args(
        args)
    weight_methods = [
        WeightMethods(args.method, n_tasks=3, device=device, **weight_methods_parameters[args.method])
        for _ in range(num_agents)
    ]
    optimizers = [
        torch.optim.Adam([
            {"params": model.parameters(), "lr": lr},
            {"params": wm.parameters(), "lr": args.method_params_lr}
        ])
        for model, wm in zip(models, weight_methods)
    ]

    # Training Loop
    scheduler = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=100, gamma=0.5)
    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    conf_mat = ConfMatrix(models[0].segnet.class_nb)
    custom_step = -1
    deltas = np.zeros([epochs,], dtype=np.float32)

    # some extra statistics we save during training
    loss_list = []
    epoch_values = []
    # Get the current time
    now = datetime.datetime.now()
    # Format the time as hhmmss
    timestamp = now.strftime("%H%M%S")

    for epoch in epoch_iter:
        t0 = time.time()
        cost = np.zeros(24, dtype=np.float32)
        train_conf_mat = ConfMatrix(models[0].segnet.class_nb)

        # Multi-agent training phase
        for batches in zip(*train_loaders):
            custom_step += 1
            all_gradients = []
            batch_losses = []

            # Compute gradients for each agent
            for agent_id, (model, weight_method, optimizer, batch) in enumerate(zip(models, weight_methods, optimizers, batches)):
                train_data, train_label, train_depth, train_normal = batch
                train_data, train_label = train_data.to(device), train_label.long().to(
                    device
                )
                train_depth, train_normal = train_depth.to(
                    device), train_normal.to(device)
                
                optimizer.zero_grad()

                # Forward pass
                train_pred, features = model(
                    train_data, return_representation=True)
                losses = torch.stack(
                    (
                        calc_loss(train_pred[0], train_label, "semantic"),
                        calc_loss(train_pred[1], train_depth, "depth"),
                        calc_loss(train_pred[2], train_normal, "normal"),
                    )
                )
                batch_losses.append(losses.detach().cpu().numpy())

                # Compute weighted loss
                loss, extra_outputs = weight_method.backward(
                    losses=losses,
                    shared_parameters=list(model.shared_parameters()),
                    task_specific_parameters=list(
                        model.task_specific_parameters()),
                    last_shared_parameters=list(model.last_shared_parameters()),
                    representation=features,
                )
                loss.backward(retain_graph=True) 

                # Store gradients
                gradients = [param.grad.clone() for param in model.parameters()]
                all_gradients.append(gradients)
                model.zero_grad()

                # Update training metrics
                train_conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())
                cost[0] += losses[0].item() / (len(train_loaders[0]) * num_agents)
                cost[3] += losses[1].item() / (len(train_loaders[0]) * num_agents)
                cost[4] += depth_error(train_pred[1], train_depth)[0] / (len(train_loaders[0]) * num_agents)
                cost[5] += depth_error(train_pred[1], train_depth)[1] / (len(train_loaders[0]) * num_agents)
                cost[6] += losses[2].item() / (len(train_loaders[0]) * num_agents)
                normal_err = normal_error(train_pred[2], train_normal)
                for i, val in enumerate(normal_err):
                    cost[7 + i] += val / (len(train_loaders[0]) * num_agents)

            #  Gradient aggregation using matrix W
            aggregated_grads = []
            for param_grads in zip(*all_gradients):
                stacked_grads = torch.stack(param_grads)
                aggregated = torch.einsum('ij,j...->i...', W_tensor, stacked_grads)
                aggregated_grads.append(aggregated)

            # Manual parameter update
            learning_rate = lr  # Use learning rate from args
            for agent_id, model in enumerate(models):
                for param, aggr_grad in zip(model.parameters(), aggregated_grads):
                    param.data -= learning_rate * aggr_grad[agent_id]
            # Store batch losses
            loss_list.extend(batch_losses)

        # Update training metrics
        avg_cost[epoch, :12] = cost[:12]
        avg_cost[epoch, 1:3] = train_conf_mat.get_metrics()

        # Parameter consensus after each epoch
        all_params = [list(model.parameters()) for model in models]
        for param_idx in range(len(all_params[0])):
            params = torch.stack([model_params[param_idx].data for model_params in all_params])
            aggr_params = torch.einsum('ij,j...->i...', W_tensor, params)
            for agent_id, model in enumerate(models):
                list(model.parameters())[param_idx].data.copy_(aggr_params[agent_id])

        # Testing Phase
        test_conf_mat = ConfMatrix(models[0].segnet.class_nb)
        test_cost = np.zeros(12, dtype=np.float32)  # Only test metrics (indices 12-23)        
        for agent_id, (model, test_loader) in enumerate(zip(models, test_loaders)):
            model.eval()
            with torch.no_grad():
                for test_data, test_label, test_depth, test_normal in test_loader:
                    test_data = test_data.to(device)
                    test_label = test_label.long().to(device)
                    test_depth = test_depth.to(device)
                    test_normal = test_normal.to(device)

                    test_pred = model(test_data)

                    # Test loss calculation
                    test_loss = torch.stack((
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                        calc_loss(test_pred[2], test_normal, "normal"),
                    ))

                    # Update test metrics
                    test_conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())
                    test_cost[0] += test_loss[0].item() / (len(test_loader) * num_agents)
                    test_cost[3] += test_loss[1].item() / (len(test_loader) * num_agents)
                    test_cost[4] += depth_error(test_pred[1], test_depth)[0] / (len(test_loader) * num_agents)
                    test_cost[5] += depth_error(test_pred[1], test_depth)[1] / (len(test_loader) * num_agents)
                    test_cost[6] += test_loss[2].item() / (len(test_loader) * num_agents)
                    normal_err = normal_error(test_pred[2], test_normal)
                    for i, val in enumerate(normal_err):
                        test_cost[7 + i] += val / (len(test_loader) * num_agents)

        # Store test metrics
        avg_cost[epoch, 12:] = test_cost
        avg_cost[epoch, 13:15] = test_conf_mat.get_metrics()

        # Calculate delta_m
        test_delta_m = delta_fn(avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]])
        deltas[epoch] = test_delta_m

        # Logging
        print(
            f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
            f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30 | ∆m (test)"
        )
        print(
            f"Epoch: {epoch:04d} | lr={lr} | method {args.method} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
            f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
            f"{avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f} || "
            f"TEST: {avg_cost[epoch, 12]:.4f} {avg_cost[epoch, 13]:.4f} {avg_cost[epoch, 14]:.4f} | "
            f"{avg_cost[epoch, 15]:.4f} {avg_cost[epoch, 16]:.4f} {avg_cost[epoch, 17]:.4f} | "
            f"{avg_cost[epoch, 18]:.4f} {avg_cost[epoch, 19]:.4f} {avg_cost[epoch, 20]:.4f} {avg_cost[epoch, 21]:.4f} {avg_cost[epoch, 22]:.4f} {avg_cost[epoch, 23]:.4f} "
            f"| {test_delta_m:.3f}"
        )
        formatted_values = f"{avg_cost[epoch, 12]:.4f} {avg_cost[epoch, 15]:.4f} {avg_cost[epoch, 18]:.4f}"

        # Append the formatted string to your list
        epoch_values.append(formatted_values)

        if wandb.run is not None:
            wandb.log(
                {"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
            wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
            wandb.log(
                {"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
            wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
            wandb.log(
                {"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
            wandb.log(
                {"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)
            wandb.log(
                {"Train Normal Loss": avg_cost[epoch, 6]}, step=epoch)
            wandb.log({"Train Loss Mean": avg_cost[epoch, 7]}, step=epoch)
            wandb.log({"Train Loss Med": avg_cost[epoch, 8]}, step=epoch)
            wandb.log(
                {"Train Loss <11.25": avg_cost[epoch, 9]}, step=epoch)
            wandb.log(
                {"Train Loss <22.5": avg_cost[epoch, 10]}, step=epoch)
            wandb.log({"Train Loss <30": avg_cost[epoch, 11]}, step=epoch)

            wandb.log(
                {"Test Semantic Loss": avg_cost[epoch, 12]}, step=epoch)
            wandb.log({"Test Mean IoU": avg_cost[epoch, 13]}, step=epoch)
            wandb.log(
                {"Test Pixel Accuracy": avg_cost[epoch, 14]}, step=epoch)
            wandb.log({"Test Depth Loss": avg_cost[epoch, 15]}, step=epoch)
            wandb.log(
                {"Test Absolute Error": avg_cost[epoch, 16]}, step=epoch)
            wandb.log(
                {"Test Relative Error": avg_cost[epoch, 17]}, step=epoch)
            wandb.log(
                {"Test Normal Loss": avg_cost[epoch, 18]}, step=epoch)
            wandb.log({"Test Loss Mean": avg_cost[epoch, 19]}, step=epoch)
            wandb.log({"Test Loss Med": avg_cost[epoch, 20]}, step=epoch)
            wandb.log(
                {"Test Loss <11.25": avg_cost[epoch, 21]}, step=epoch)
            wandb.log({"Test Loss <22.5": avg_cost[epoch, 22]}, step=epoch)
            wandb.log({"Test Loss <30": avg_cost[epoch, 23]}, step=epoch)
            wandb.log({"Test ∆m": test_delta_m}, step=epoch)

        keys = [
            "Train Semantic Loss",
            "Train Mean IoU",
            "Train Pixel Accuracy",
            "Train Depth Loss",
            "Train Absolute Error",
            "Train Relative Error",
            "Train Normal Loss",
            "Train Loss Mean",
            "Train Loss Med",
            "Train Loss <11.25",
            "Train Loss <22.5",
            "Train Loss <30",

            "Test Semantic Loss",
            "Test Mean IoU",
            "Test Pixel Accuracy",
            "Test Depth Loss",
            "Test Absolute Error",
            "Test Relative Error",
            "Test Normal Loss",
            "Test Loss Mean",
            "Test Loss Med",
            "Test Loss <11.25",
            "Test Loss <22.5",
            "Test Loss <30"
        ]

        name = f"{args.method}_sd{args.seed}_lr{args.lr}_bs{args.batch_size}_ng{args.num_agents}_sp{args.network_sparsity}_{timestamp}"

        torch.save({
            "delta_m": deltas,
            "keys": keys,
            "avg_cost": avg_cost,
            "losses": loss_list,
        }, f"./save/{name}.stats")

        scheduler.step()

    txt_name = os.path.join(os.getcwd(), f"trainlogs/losses/{name}.txt")
    with open(txt_name, 'w') as file:
        for value in epoch_values:
            file.write(value + '\n')

if __name__ == "__main__":
    parser = ArgumentParser("NYUv2", parents=[common_parser])
    parser.add_argument("--num_agents", type=int, default=5,
                      help="Number of collaborative agents")
    parser.add_argument("--network_sparsity", type=float, default=0.5,
                      help="Sparsity level of communication network (0-1)")
    
    # Original arguments
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=2,
        batch_size=2,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mtan",
        choices=["segnet", "mtan"],
        help="model type",
    )
    parser.add_argument(
        "--apply-augmentation", type=str2bool, default=True, help="data augmentations"
    )
    parser.add_argument("--wandb_project", type=str,
                        default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Name of Weights & Biases Entity.")
    args = parser.parse_args()
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity, config=args)

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)

    if wandb.run is not None:
        wandb.finish()