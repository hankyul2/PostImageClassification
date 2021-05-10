import os

import numpy as np
import pytest
import torch

from src.gpu import GPU
from src.utils.config import get_config
from src.utils.data import download_preprocess_save_data
from src.utils.post_train import make_input, get_dataset
from src.vanila_scafolding import get_model, get_data


def test_make_input_create(args):
    gpu = GPU()
    device = gpu.get_device(args.gpu)
    download_preprocess_save_data()

    config = get_config()
    dataset_cfg = config[args.dataset]
    shared_cfg = config["shared"]
    alg_cfg = config[args.alg]

    base_model_path = os.path.join(args.output, args.base_model_name, 'best_model.pth')
    post_data_root = os.path.join('data/cifar10', args.base_model_name)
    if not os.path.exists(post_data_root):
        os.makedirs(post_data_root)

    l_train_dataset, u_train_dataset, val_dataset, test_dataset = get_dataset(args, dataset_cfg)
    model, optimizer, ssl_obj = get_model(args, alg_cfg, dataset_cfg, device)
    model.load_state_dict(torch.load(base_model_path))

    label_train_path = os.path.join(post_data_root, '{}.npy'.format('l_train'))
    label_test_path = os.path.join(post_data_root, '{}.npy'.format('test'))
    label_valid_path = os.path.join(post_data_root, '{}.npy'.format('val'))

    make_input(model, l_train_dataset, label_train_path, device)
    make_input(model, val_dataset, label_valid_path, device)
    make_input(model, test_dataset, label_test_path, device)

    assert os.path.exists(label_train_path)
    assert os.path.exists(label_test_path)
    assert os.path.exists(label_valid_path)

    label_train_input = np.load(label_train_path, allow_pickle=True).item()
    label_valid_input = np.load(label_valid_path, allow_pickle=True).item()
    label_test_input = np.load(label_test_path, allow_pickle=True).item()

    assert len(label_train_input['images']) == 4000
    assert len(label_valid_input['images']) == 5000
    assert len(label_test_input['images']) == 10000




