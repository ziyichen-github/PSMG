from methods.weight_methods import WeightMethods
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from experiments.celeba.models import Network
from experiments.celeba.data import CelebaDataset
# import os
from argparse import ArgumentParser

import numpy as np
import time
# import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
_ = torch.cuda.empty_cache()


class CelebaMetrics():
    """
    CelebA metric accumulator.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def incr(self, y_preds, ys):
        # y_preds: [ y_pred (batch, 1) ] x 40
        # ys     : [ y_pred (batch, 1) ] x 40
        y_preds = torch.stack(y_preds).detach()  # (40, batch, 1)
        ys = torch.stack(ys).detach()      # (40, batch, 1)
        y_preds = y_preds.gt(0.5).float()
        self.tp += (y_preds * ys).sum([1, 2])  # (40,)
        self.fp += (y_preds * (1 - ys)).sum([1, 2])
        self.fn += ((1 - y_preds) * ys).sum([1, 2])

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.cpu().numpy()


def main(path, lr, bs, device):
    # we only train for specific task
    model = Network().to(device)

    train_set = CelebaDataset(data_dir=path, split='train')
    val_set = CelebaDataset(data_dir=path, split='val')
    test_set = CelebaDataset(data_dir=path, split='test')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=bs, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = args.n_epochs

    metrics = np.zeros([epochs, 40], dtype=np.float32)  # test_f1
    metric = CelebaMetrics()
    loss_fn = torch.nn.BCELoss()
    epoch_values = []
    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(
        args)
    weight_method = WeightMethods(
        args.method, n_tasks=40, device=device, **weight_methods_parameters[args.method]
    )

    best_val_f1 = 0.0
    best_epoch = None
    epoch_len = len(train_loader)
    now = datetime.datetime.now()
    timestamp = now.strftime("%H%M%S")
    for epoch in range(epochs):
        # training
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            y_ = model(x)
            losses = torch.stack([loss_fn(y_task_pred, y_task)
                                 for (y_task_pred, y_task) in zip(y_, y)])
            optimizer.zero_grad()
            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(
                    model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
            )
            optimizer.step()
            # if "famo" in args.method:
            if args.method in ["famo", "pmgdn", "pmgdnlog"]:
                with torch.no_grad():
                    y_ = model(x)
                    new_losses = torch.stack(
                        [loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                    weight_method.method.update(new_losses.detach())
        t1 = time.time()

        model.eval()
        # validation
        metric.reset()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([loss_fn(y_task_pred, y_task)
                                     for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
        val_f1 = metric.result()
        if val_f1.mean() > best_val_f1:
            best_val_f1 = val_f1.mean()
            best_epoch = epoch

        # testing
        metric.reset()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([loss_fn(y_task_pred, y_task)
                                     for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
        test_f1 = metric.result()
        metrics[epoch] = test_f1
        epoch_values.append(round(test_f1.mean(), 5))
        t2 = time.time()
        print(
            f"[info] epoch {epoch} | epoch lenth={epoch_len} |train takes {(t1-t0):.2f} s | test takes {(t2-t1):.2f} s | val_f1 {val_f1.mean():.4f} | test_f1 {test_f1.mean():.4f} | Best {best_epoch}")
        if "famo" in args.method:
            name = f"{args.method}_gamma{args.gamma}_sd{args.seed}_{timestamp}"
        else:
            name = f"{args.method}_sd{args.seed}_{timestamp}"

        torch.save({"metric": metrics, "best_epoch": best_epoch},
                   f"./save/{name}.stats")
    txt_name = f"/home/mx6835/Academic/MM1204/FAMO/experiments/celeba/trainlogs/losses/{name}.txt"
    with open(txt_name, 'w') as file:
        for value in epoch_values:
            file.write(str(value) + '\n')
    print("Training finished.", epoch_values)


if __name__ == "__main__":
    parser = ArgumentParser("Celeba", parents=[common_parser])
    parser.set_defaults(
        # data_path=os.path.join(os.getcwd(), "dataset"),
        data_path="/home/mx6835/Academic/MM1204/unitary-scalarization-dmtl/data/celeba",
        lr=3e-4,
        n_epochs=15,
        batch_size=256,
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    main(path=args.data_path,
         lr=args.lr,
         bs=args.batch_size,
         device=device)

# Single task baseline:
# [0.6736886  0.68121034 0.81524944 0.5760289  0.7205613  0.8555076
#  0.38203922 0.58225113 0.787647   0.8321292  0.5029583  0.68694085
#  0.6781237  0.5240381  0.5161666  0.95694304 0.6968786  0.67976356
#  0.8808315  0.8582131  0.97034    0.93267566 0.5057539  0.40307626
#  0.9703734  0.48644206 0.60786104 0.5261031  0.56907415 0.59815097
#  0.6858371  0.924108   0.5424991  0.7406311  0.71019936 0.87365365
#  0.9305602  0.33704284 0.7647628  0.91907   ]


#  [6.78580165e-01, 5.29595017e-01, 8.16967547e-01, 4.95214880e-01,
    # 7.42459416e-01, 8.42832446e-01, 3.25399548e-01, 5.75024188e-01,
    # 7.92373300e-01, 8.15462828e-01, 2.95707434e-01, 6.81865811e-01,
    # 6.30941272e-01, 5.38583875e-01, 5.35519183e-01, 9.53678429e-01,
    # 6.36024892e-01, 6.67889893e-01, 8.72464359e-01, 8.59030843e-01,
    # 9.73812461e-01, 9.25329328e-01, 5.30898809e-01, 3.04988354e-01,
    # 9.71670866e-01, 4.43082333e-01, 4.84502435e-01, 3.78164202e-01,
    # 5.24601996e-01, 6.25444233e-01, 6.74060345e-01, 9.13077712e-01,
    # 3.81181329e-01, 7.43346214e-01, 6.81629479e-01, 8.80533636e-01,
    # 9.30246294e-01, 1.96032688e-01, 7.44294047e-01, 9.17772353e-01],

# [0.65800625, 0.64952344, 0.81971484, 0.5919729 , 0.72891563,
#         0.86124706, 0.49341577, 0.5905738 , 0.7157014 , 0.8334559 ,
#         0.5299748 , 0.6655496 , 0.6793376 , 0.46642682, 0.45142   ,
#         0.9530942 , 0.6908703 , 0.6385069 , 0.88461065, 0.8592406 ,
#         0.97382337, 0.92676383, 0.42617154, 0.40411726, 0.97088563,
#         0.45892975, 0.6089626 , 0.44013876, 0.46295556, 0.6397735 ,
#         0.718363  , 0.92070687, 0.53057   , 0.6574741 , 0.68851155,
#         0.86890954, 0.9347012 , 0.24882239, 0.7520001 , 0.9186254 ],
