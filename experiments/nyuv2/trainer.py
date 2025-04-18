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
# import sys
# sys.path.append("/home/mx6835/Academic/MM1204/UniversalRepresentations/DensePred/data")

import numpy as np
import torch
import time
import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
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
    # ----
    # Nets
    # ---
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    nyuv2_train_set = NYUv2(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    nyuv2_test_set = NYUv2(root=path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=bs, shuffle=True
    )
    train_aux_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=bs, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set, batch_size=bs, shuffle=False
    )

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(
        args)

    weight_method = WeightMethods(
        args.method, n_tasks=3, device=device, **weight_methods_parameters[args.method]
    )

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
            dict(params=weight_method.parameters(), lr=args.method_params_lr),
        ],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5)
    epoch_len = len(train_loader)
    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)
    deltas = np.zeros([epochs,], dtype=np.float32)

    # some extra statistics we save during training
    loss_list = []
    epoch_values = []
    # Get the current time
    now = datetime.datetime.now()
    # Format the time as hhmmss
    timestamp = now.strftime("%H%M%S")
    # filename = f"{args.method}_gamma{args.gamma}_sd{args.seed}_lr{args.lr}_bs{args.batch_size}_{timestamp}.txt"
    for epoch in epoch_iter:
        t0 = time.time()
        cost = np.zeros(24, dtype=np.float32)

        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()
            optimizer.zero_grad()
            train_data, train_label, train_depth, train_normal = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth, train_normal = train_depth.to(
                device), train_normal.to(device)

            train_pred, features = model(
                train_data, return_representation=True)

            if "sdmgrad" in args.method:
                aux_batch1 = next(iter(train_aux_loader))
                train_aux_data1, train_aux_label1, train_aux_depth1, train_aux_normal1 = aux_batch1
                train_aux_data1, train_aux_label1 = train_aux_data1.to(
                    device), train_aux_label1.long().to(device)
                train_aux_depth1, train_aux_normal1 = train_aux_depth1.to(
                    device), train_aux_normal1.to(device)
                train_aux_pred1, aux_features1 = model(
                    train_aux_data1, return_representation=True)

                aux_batch2 = next(iter(train_aux_loader))
                train_aux_data2, train_aux_label2, train_aux_depth2, train_aux_normal2 = aux_batch2
                train_aux_data2, train_aux_label2 = train_aux_data2.to(
                    device), train_aux_label2.long().to(device)
                train_aux_depth2, train_aux_normal2 = train_aux_depth2.to(
                    device), train_aux_normal2.to(device)
                train_aux_pred2, aux_features2 = model(
                    train_aux_data2, return_representation=True)

                losses = torch.stack(
                    (
                        calc_loss(train_pred[0], train_label, "semantic"),
                        calc_loss(train_pred[1], train_depth, "depth"),
                        calc_loss(train_pred[2], train_normal, "normal"),
                        calc_loss(train_aux_pred1[0],
                                  train_aux_label1, "semantic"),
                        calc_loss(train_aux_pred1[1],
                                  train_aux_depth1, "depth"),
                        calc_loss(train_aux_pred1[2],
                                  train_aux_normal1, "normal"),
                        calc_loss(train_aux_pred2[0],
                                  train_aux_label2, "semantic"),
                        calc_loss(train_aux_pred2[1],
                                  train_aux_depth2, "depth"),
                        calc_loss(train_aux_pred2[2],
                                  train_aux_normal2, "normal"),
                    )
                )
            else:
                losses = torch.stack(
                    (
                        calc_loss(train_pred[0], train_label, "semantic"),
                        calc_loss(train_pred[1], train_depth, "depth"),
                        calc_loss(train_pred[2], train_normal, "normal"),
                    )
                )

            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(
                    model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )

            # for record intermediate statistics
            loss_list.append(losses.detach().cpu())
            optimizer.step()
            # if "famo" in args.method:
            if args.method in ["famo", "pmgdn", "pmgdnlog"]:
                with torch.no_grad():
                    train_pred = model(train_data, return_representation=False)
                    new_losses = torch.stack(
                        (
                            calc_loss(train_pred[0], train_label, "semantic"),
                            calc_loss(train_pred[1], train_depth, "depth"),
                            calc_loss(train_pred[2], train_normal, "normal"),
                        )
                    )
                    weight_method.method.update(new_losses.detach())

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(
                1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = losses[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(
                train_pred[2], train_normal
            )
            avg_cost[epoch, :12] += cost[:12] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
                f"normal loss: {losses[2].item():.3f}"
            )
        t1 = time.time()
        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # todo: move evaluate to function?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth, test_normal = test_depth.to(
                    device), test_normal.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                        calc_loss(test_pred[2], test_normal, "normal"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(
                    1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(
                    test_pred[2], test_normal
                )
                avg_cost[epoch, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 13:15] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]]
            )
            deltas[epoch] = test_delta_m

            # print results
            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
                f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30 | ∆m (test)"
            )
            print(
                f"Epoch: {epoch:04d} | lr={lr} | epoch lenth={epoch_len} | train time: {t1-t0:.3f} s | method {args.method} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
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

            # if "famo" in args.method:
            if args.method in ["famo", "pmgdn", "pmgdnlog"]:
                name = f"{args.method}_gamma{args.gamma}_sd{args.seed}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
            elif "fairgrad" in args.method:
                name = f"{args.method}_alpha{args.alpha}_sd{args.seed}_{timestamp}"
            else:
                name = f"{args.method}_sd{args.seed}_lr{args.lr}_bs{args.batch_size}_{timestamp}"

            torch.save({
                "delta_m": deltas,
                "keys": keys,
                "avg_cost": avg_cost,
                "losses": loss_list,
            }, f"./save/{name}.stats")
    txt_name = os.path.join(os.getcwd(), f"trainlogs/losses/{name}.txt")
    with open(txt_name, 'w') as file:
        for value in epoch_values:
            file.write(value + '\n')
    # print("Training finished.", epoch_values)


if __name__ == "__main__":
    parser = ArgumentParser("NYUv2", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=200,
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

    # set seed
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity, config=args)

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)

    if wandb.run is not None:
        wandb.finish()
