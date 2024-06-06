from argparse import ArgumentParser
import os
import time
import datetime
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import trange
import wandb

from experiments.quantum_chemistry.models import Net
from experiments.quantum_chemistry.utils import (
    Complete,
    MyTransform,
    delta_fn,
    multiply_indx,
)
from experiments.quantum_chemistry.utils import target_idx as targets
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from methods.weight_methods import WeightMethods

set_logger()

# Cannot use ACC
# https://blog.csdn.net/KPer_Yang/article/details/129105477


@torch.no_grad()
def evaluate(model, loader, std, scale_target):
    model.eval()
    data_size = 0.0
    task_losses = 0.0
    # correct_predictions = 0

    for i, data in enumerate(loader):
        data = data.to(device)
        out = model(data)

        if scale_target:
            task_losses += F.l1_loss(out * std.to(device), data.y *
                                     std.to(device), reduction="none").sum(0)  # MAE
        else:
            task_losses += F.l1_loss(out, data.y,
                                     reduction="none").sum(0)  # MAE

        data_size += len(data.y)

        # Convert the predicted values to the same data type as the target values
        # predicted = out.detach().round().type_as(data.y)
        # correct_predictions += (predicted == data.y).sum().item()

    model.train()
    avg_task_losses = task_losses / data_size  # Report meV instead of eV.
    avg_task_losses = avg_task_losses.detach().cpu().numpy()
    # here we get the loss array and delta_m
    avg_task_losses[multiply_indx] *= 1000
    delta_m = delta_fn(avg_task_losses)
    # accuracy = correct_predictions / data_size

    return dict(
        avg_loss=avg_task_losses.mean(),
        avg_task_losses=avg_task_losses,
        delta_m=delta_m,
        # avg_acc=accuracy,
    )


def main(
    data_path: str,
    batch_size: int,
    device: torch.device,
    method: str,
    weight_method_params: dict,
    lr: float,
    method_params_lr: float,
    n_epochs: int,
    targets: list = None,
    scale_target: bool = True,
    main_task: int = None,
):
    dim = 64
    model = Net(n_tasks=len(targets), num_features=11, dim=dim).to(device)

    transform = T.Compose(
        [MyTransform(targets), Complete(), T.Distance(norm=False)])
    dataset = QM9(data_path, transform=transform).shuffle()

    # Split datasets.
    test_dataset = dataset[:10000]
    val_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:]

    std = None
    if scale_target:
        mean = train_dataset.data.y[:, targets].mean(dim=0, keepdim=True)
        std = train_dataset.data.y[:, targets].std(dim=0, keepdim=True)

        dataset.data.y[:, targets] = (dataset.data.y[:, targets] - mean) / std

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    weight_method = WeightMethods(
        method,
        n_tasks=len(targets),
        device=device,
        **weight_method_params[method],
    )

    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
            dict(params=weight_method.parameters(), lr=method_params_lr),
        ],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=4, min_lr=0.00002
    )

    epoch_iterator = trange(n_epochs)

    best_val = np.inf
    best_test = np.inf
    best_test_delta = np.inf
    best_val_delta = np.inf
    best_test_results = None
    best_epoch = 0

    avg_cost = np.zeros([n_epochs, 13*2], dtype=np.float32)
    deltas = np.zeros([n_epochs], dtype=np.float32)

    # some extra statistics we save during training
    loss_list = []
    epoch_len = len(train_loader)
    # Get the current time
    now = datetime.datetime.now()
    # Format the time as hhmmss
    timestamp = now.strftime("%H%M%S")
    test_losses = []
    for epoch in epoch_iterator:
        t0 = time.time()
        lr = scheduler.optimizer.param_groups[0]["lr"]
        # lenth of batchs
        # batch_len = len(train_loader)
        for j, data in enumerate(train_loader):
            model.train()

            data = data.to(device)
            optimizer.zero_grad()

            out, features = model(data, return_representation=True)

            losses = F.mse_loss(out, data.y, reduction="none").mean(0)

            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(
                    model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )

            loss_list.append(losses.detach().cpu())
            optimizer.step()

            # if "famo" in args.method:
            if args.method in ["famo", "pmgdn", "pmgdnlog"]:
                # if "famo" or "pmgd" in args.method:
                with torch.no_grad():
                    out_ = model(data, return_representation=False)
                    new_losses = F.mse_loss(
                        out_, data.y, reduction="none").mean(0)
                    weight_method.method.update(new_losses.detach())
        t1 = time.time()

        val_loss_dict = evaluate(
            model, val_loader, std=std, scale_target=scale_target)
        test_loss_dict = evaluate(
            model, test_loader, std=std, scale_target=scale_target
        )

        val_loss = val_loss_dict["avg_loss"]
        val_delta = val_loss_dict["delta_m"]
        test_loss = test_loss_dict["avg_loss"]
        test_delta = test_loss_dict["delta_m"]
        test_losses.append(test_loss)

        if method == "stl":
            best_val_criteria = val_loss_dict["avg_task_losses"][main_task] <= best_val
        else:
            best_val_criteria = val_delta <= best_val_delta

        if best_val_criteria:  # It use val_delta to determine the best model
            best_val = val_loss
            best_test = test_loss
            best_test_results = test_loss_dict
            best_val_delta = val_delta
            best_test_delta = test_delta
            best_epoch = epoch

        avg_cost[epoch, 0] = val_loss
        avg_cost[epoch, 1] = val_delta
        avg_cost[epoch, 2:2+11] = val_loss_dict["avg_task_losses"]
        avg_cost[epoch, 13] = test_loss
        avg_cost[epoch, 14:14+11] = test_loss_dict["avg_task_losses"]
        deltas[epoch] = test_delta

        # for logger
        epoch_iterator.set_description(
            f"epoch {epoch} | lr={lr} | epoch lenth={epoch_len} | train time: {t1-t0:.3f} s | method {args.method} | train loss {losses.mean().item():.3f} | val loss: {val_loss:.3f} | "
            f"test loss: {test_loss:.3f} | best test loss {best_test:.3f} | best_test_delta {best_test_delta:.3f} | best epoch {best_epoch}"
        )

        if wandb.run is not None:
            wandb.log({"Learning Rate": lr}, step=epoch)
            wandb.log({"Train Loss": losses.mean().item()}, step=epoch)
            wandb.log({"Val Loss": val_loss}, step=epoch)
            wandb.log({"Val Delta": val_delta}, step=epoch)
            wandb.log({"Test Loss": test_loss}, step=epoch)
            wandb.log({"Test Delta": test_delta}, step=epoch)
            wandb.log({"Best Test Loss": best_test}, step=epoch)
            wandb.log({"Best Test Delta": best_test_delta}, step=epoch)

        scheduler.step(
            val_loss_dict["avg_task_losses"][main_task]
            if method == "stl"
            else val_delta
        )

        if "famo" in args.method:
            if args.scale_y:
                name = f"{args.method}_gamma{args.gamma}_wlr{args.method_params_lr}_scale_sd{args.seed}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
            else:
                name = f"{args.method}_gamma{args.gamma}_wlr{args.method_params_lr}_sd{args.seed}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
        elif "fairgrad" in args.method:
            if args.scale_y:
                name = f"{args.method}_alpha{args.alpha}_scale_sd{args.seed}"
            else:
                name = f"{args.method}_alpha{args.alpha}_sd{args.seed}"
        elif "stl" in args.method:
            name = f"{args.method}_task{args.main_task}_sd{args.seed}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
        else:
            name = f"{args.method}_sd{args.seed}_lr{args.lr}_bs{args.batch_size}_{timestamp}"

        torch.save({
            "avg_cost": avg_cost,
            "losses": loss_list,
            "delta_m": deltas,
        }, f"./save/{name}.stats")
        txt_name = f"/home/mx6835/Academic/MM1204/FAMO/experiments/quantum_chemistry/trainlogs/losses/{name}.txt"
        with open(txt_name, 'w') as f:
            losses_string = "\n".join(str(loss) for loss in test_losses)
            f.write(losses_string)


if __name__ == "__main__":
    _ = torch.cuda.empty_cache()
    parser = ArgumentParser("QM9", parents=[common_parser])
    parser.set_defaults(
        # data_path=os.path.join(os.getcwd(), "dataset"),
        data_path="/home/mx6835/Academic/dataset4all/FAMO/QM9",
        lr=1e-3,
        n_epochs=300,
        batch_size=120,
        method="nashmtl",
    )
    parser.add_argument("--scale-y", default=False, type=str2bool)
    parser.add_argument("--wandb_project", type=str,
                        default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Name of Weights & Biases Entity.")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity, config=args)

    weight_method_params = extract_weight_method_parameters_from_args(args)

    device = get_device(gpus=args.gpu)
    main(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device,
        method=args.method,
        weight_method_params=weight_method_params,
        lr=args.lr,
        method_params_lr=args.method_params_lr,
        n_epochs=args.n_epochs,
        targets=targets,
        scale_target=args.scale_y,
        main_task=args.main_task,
    )

    if wandb.run is not None:
        wandb.finish()
