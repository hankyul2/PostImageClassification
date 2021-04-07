import pytest

from src.gpu import GPUChecker


def test_gpu_checker_construction():
    g = GPUChecker(0)
    assert g.gpu_id == 0

def test_gpu_checker_check_maximum():
    g = GPUChecker(0)
    with pytest.raises(AssertionError) as exc:
        g.check_gpu(25000)
    assert str(exc.value) == 'AssertionError occur'

def test_gpu_checker_check_minimum():
    g = GPUChecker(0)
    with pytest.raises(AssertionError) as exc:
        g.check_gpu(-1)
    assert str(exc.value) == 'AssertionError occur'



