import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def make_input(net, ds, output_path, device=None, save=True):
    if os.path.exists(output_path):
        return

    net.eval()

    class IdentityLayer(nn.Module):
        def forward(self, x):
            return x

    net.output = IdentityLayer()
    output = []
    label = []

    if device:
        net = net.to(torch.device(device))

    dl = DataLoader(ds, batch_size=100)

    for (images, labels) in dl:

        if device:
            images = images.to(device=device)
            labels = labels.to(device=device)

        images, labels = images.float(), labels.long()

        outputs = net(images)
        output += outputs.detach().tolist()
        label += labels.detach().tolist()
    if save:
        post_data = {"images": np.array(output), "labels": np.array(label)}
        np.save(output_path, post_data)

def get_dataset(args, dataset_cfg):
    l_train_dataset = dataset_cfg["dataset"](args.root, "l_train")
    u_train_dataset = dataset_cfg["dataset"](args.root, "u_train")
    val_dataset = dataset_cfg["dataset"](args.root, "val")
    test_dataset = dataset_cfg["dataset"](args.root, "test")

    return l_train_dataset, u_train_dataset, val_dataset, test_dataset

