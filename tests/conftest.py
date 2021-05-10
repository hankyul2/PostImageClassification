import GPUtil
import pytest

from easydict import EasyDict as edict

from src.gpu import GPU
from src.monitor import Monitor


@pytest.fixture
def gpu() -> GPU:
    g = GPU()
    return g


@pytest.fixture
def gpu_id() -> int:
    gpu_id = GPUtil.getFirstAvailable()
    return gpu_id


@pytest.fixture
def monitor() -> Monitor:
    gpu_id = GPUtil.getFirstAvailable()
    m = Monitor(10, gpu_id)
    return m

@pytest.fixture
def args() -> edict:
    d = edict({
        'alg': "PL",
        'em': 0,
        'validation': 820,
        'dataset': "cifar10",
        'root': "data",
        'model_name':'pl1',
        'base_model_name':'cifar10_PL_1620253529',
        'output': "../exp_res",
        'gpu': 3
    })
    return d