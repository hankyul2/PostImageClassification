import GPUtil
import pytest

from src.gpu import GPU
from src.monitor import Monitor


@pytest.fixture
def gpu() -> GPU:
    g = GPU()
    return g


@pytest.fixture
def monitor() -> Monitor:
    gpu_id = GPUtil.getFirstAvailable()
    m = Monitor(10, gpu_id)
    return m
