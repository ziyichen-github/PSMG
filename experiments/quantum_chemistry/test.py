import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Sample model and additional method for demonstration purposes


class SampleModel(torch.nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        return torch.nn.functional.relu(self.conv2(x))


class WeightMethod(torch.nn.Module):
    def __init__(self):
        super(WeightMethod, self).__init__()
        self.dense = torch.nn.Linear(20, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.dense(x))


model = SampleModel()
weight_method = WeightMethod()

# Setup: Optimizer
lr_main = 0.001  # Base learning rate for the main model
lr_method = 0.01  # Base learning rate for the weight method

optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': lr_main},
    {'params': weight_method.parameters(), 'lr': lr_method}
])

# Setup: Learning Rate Schedulers

# Assuming you have some train_loader defined somewhere that feeds data into the model
train_loader = ...  # Placeholder for the training data loader

# Warm-up Scheduler
warmup_factor = 1.0 / 3
warmup_iters = 5 * len(train_loader)  # 5 epochs worth of warm-up

scheduler_warmup = lr_scheduler.LinearLR(
    optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

# Cyclic Scheduler
# Lower boundary of learning rate (can be same as lr_main or different)
base_lr = 0.001
# Upper boundary of learning rate (can be same as lr_method or different)
max_lr = 0.01

scheduler_cyclic = lr_scheduler.CyclicLR(optimizer,
                                         base_lr=base_lr,
                                         max_lr=max_lr,
                                         step_size_up=5 * len(train_loader),
                                         mode='triangular2')

# ReduceLROnPlateau Scheduler
scheduler_plateau = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.7,
    patience=5,
    min_lr=0.00001
)

# Combining Schedulers: Sequentially applying warmup, then cyclic, and finally ReduceLROnPlateau
scheduler_seq = lr_scheduler.SequentialLR(optimizer,
                                          schedulers=[
                                              scheduler_warmup, scheduler_cyclic],
                                          milestones=[warmup_iters, 100 * len(train_loader)])

# Training Loop Example


def train():
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Step: {batch_idx}\t Loss: {loss.item()}')

        # Update learning rate
        scheduler_seq.step()

    # After completing all cycles, switch to ReduceLROnPlateau
    # Assuming `val_loss` is computed after the epoch
    val_loss = compute_validation_loss()
    scheduler_plateau.step(val_loss)

    print(f'Epoch Finished: Total Loss: {total_loss}')


def compute_validation_loss():
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data, target in validation_loader:
            output = model(data)
            total_val_loss += torch.nn.functional.cross_entropy(
                output, target).item()
    return total_val_loss / len(validation_loader)


# Placeholder: You need to define `validation_loader` appropriately
validation_loader = ...

# Start Training
train()
