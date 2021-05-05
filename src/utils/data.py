import os

import numpy as np
from torchvision import datasets


def split_l_u(train_set, n_labels):
    # NOTE: this function assume that train_set is shuffled.
    images, labels = train_set.values()
    classes = np.unique(labels)
    n_labels_per_cls = n_labels // len(classes)
    l_images, l_labels, u_images, u_labels, u_real_labels = [[] for _ in range(5)]
    for c in classes:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:]]
        u_real_labels += [c_labels[n_labels_per_cls:]]
        u_labels += [np.zeros_like(c_labels[n_labels_per_cls:]) - 1] # dammy label
    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0), "real_labels": np.concatenate(u_real_labels, 0)}
    return l_train_set, u_train_set


def split_validation(train_set, validation_count):
    train_images = train_set["images"][validation_count:]
    train_labels = train_set["labels"][validation_count:]
    validation_images = train_set["images"][:validation_count]
    validation_labels = train_set["labels"][:validation_count]
    validation_set = {"images": validation_images, "labels": validation_labels}
    train_set = {"images": train_images, "labels": train_labels}
    return train_set, validation_set


def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)


def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp


def gcn(images, multiplier=55, eps=1e-10):
    # global contrast normalization
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    return multiplier * images / per_image_norm


def load_and_gcn(data_dir):
    splits = []
    for train in [True, False]:
        data_org = datasets.CIFAR10(data_dir, train, download=True)
        data_new = {"images":gcn(data_org.data), "labels":np.array(data_org.targets)}
        splits.append(data_new)
    return splits


def download_preprocess_save_data(seed=1, dataset='cifar10', nlabels=4000, data_dir='./data'):
    if os.path.exists(os.path.join(data_dir, dataset, 'test.npy')):
        return
    COUNTS = {"cifar10": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0}}
    rng = np.random.RandomState(seed)
    validation_count = COUNTS[dataset]['valid']
    if dataset == 'cifar10':
        train_set, test_set = load_and_gcn(data_dir)
        mean, zca_decomp = get_zca_normalization_param(train_set["images"])
        train_set["images"] = zca_normalization(train_set["images"], mean, zca_decomp)
        test_set["images"] = zca_normalization(test_set["images"], mean, zca_decomp)
        train_set["images"] = np.transpose(train_set["images"], (0,3,1,2))
        test_set["images"] = np.transpose(test_set["images"], (0,3,1,2))
    indices = rng.permutation(len(train_set['images']))
    train_set['images'] = train_set['images'][indices]
    train_set['labels'] = train_set['labels'][indices]
    train_set, validation_set = split_validation(train_set, validation_count)
    l_train_set, u_train_set = split_l_u(train_set, nlabels)
    if not os.path.exists(os.path.join(data_dir, dataset)):
        os.mkdir(os.path.join(data_dir, dataset))
    datasets = dict(l_train=l_train_set, u_train=u_train_set, val=validation_set, test=test_set)
    for key, value in datasets.items():
        np.save(os.path.join(data_dir, dataset, key), value)