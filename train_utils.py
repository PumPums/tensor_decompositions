import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding="",
        dilation=1,
        groups=1,
        bias=False,
        act=True,
        conv_layer=nn.Conv2d,
        core_ranks=None,
        stick_rank=None,
    ):
        super(ConvBnAct, self).__init__()
        if padding == "":
            padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        self.conv = nn.Sequential(
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                core_ranks=core_ranks,
                stick_rank=stick_rank,
            )
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act
        if self.act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x


def train_model(model, device, optimizer, dataloaders, num_epochs=5):
    """Simple training pipeline"""
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = {"train": {}, "test": {}}
    accuracy = {"train": {}, "test": {}}

    for epoch in tqdm(range(num_epochs), desc=f"Train process", leave=False):
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            total = 0
            correct = 0
            for X_batch, y_batch in tqdm(
                dataloaders[phase], leave=False, desc=f"Epoch ({phase})- {epoch + 1}"
            ):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                if phase == "train":
                    optimizer.zero_grad()

                if phase == "train":
                    y_pred = model(X_batch)
                else:
                    with torch.no_grad():
                        y_pred = model(X_batch)
                preds = torch.argmax(y_pred, -1)
                total += y_batch.size(0)
                correct += (preds == y_batch).sum().item()
                loss = loss_fn(y_pred, y_batch)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            losses[phase][epoch] = float(loss)
            accuracy[phase][epoch] = float(correct / total)
            print(
                f"Epoch-{epoch + 1} {phase} loss: {losses[phase][epoch]:.3f} \
                  <-> accuracy: {accuracy[phase][epoch]:.3f}"
            )
    return model, losses, accuracy


def eval_model(model, device, dataloader, r=False):
    """Simple eval pipeline"""
    model.eval()
    total = 0
    correct = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_pred = model(X_batch)
        preds = torch.argmax(y_pred, -1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()
    accuracy = float(correct / total)
    print(f"Model test accuracy: {accuracy}")

    if r:
        return accuracy
