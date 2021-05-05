from src.utils.dataset import CIFAR10


def get_config():
    shared_config = {
        "iteration": 500000,
        "warmup": 200000,
        "lr_decay_iter": 400000,
        "lr_decay_factor": 0.2,
        "batch_size": 100,
    }
    cifar10_config = {
        "transform": [True, True, True],
        "dataset": CIFAR10,
        "num_classes": 10,
    }
    pl_config = {
        # pseudo label
        "threashold": 0.95,
        "lr": 3e-4,
        "consis_coef": 1,
    }
    pt_config = {
        # post train
        "top_k": 5,
        "lr": 3e-4,
        "consis_coef": 1,
    }
    supervised_config = {
        "lr": 3e-3
    }
    config = {
        "shared": shared_config,
        "cifar10": cifar10_config,
        "supervised": supervised_config,
        "PL": pl_config,
        "PT": pt_config,
    }
    return config
