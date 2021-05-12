import os
import random

import numpy as np
import torch
import torch.nn.functional as F


class CIFAR10:
    def __init__(self, root, split="l_train"):
        self.split = split
        self.dataset = np.load(os.path.join(root, "cifar10", split + ".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        if self.split == 'u_train':
            real_label = self.dataset["real_labels"][idx]
            return image, label, real_label
        return image, label

    def __len__(self):
        return len(self.dataset["images"])


class POSTDATASET:
    def __init__(self, root, split="l_train"):
        self.split = split
        self.dataset = np.load(os.path.join(root, split + ".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        # if self.split == 'u_train':
        #     real_label = self.dataset["real_labels"][idx]
        #     return image, label, real_label
        return image, label

    def __len__(self):
        return len(self.dataset["images"])


class transform:
    def __init__(self, flip=True, r_crop=True, g_noise=True):
        self.flip = flip
        self.r_crop = r_crop
        self.g_noise = g_noise
        print("holizontal flip : {}, random crop : {}, gaussian noise : {}".format(
            self.flip, self.r_crop, self.g_noise
        ))

    def __call__(self, x):
        if self.flip and random.random() > 0.5:
            x = x.flip(-1)
        if self.r_crop:
            h, w = x.shape[-2:]
            x = F.pad(x, [2, 2, 2, 2], mode="reflect")
            l, t = random.randint(0, 4), random.randint(0, 4)
            x = x[:, :, t:t + h, l:l + w]
        if self.g_noise:
            n = torch.randn_like(x) * 0.15
            x = n + x
        return x


class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """

    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
