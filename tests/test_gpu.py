import pytest


def test_gpu_construction(gpu):
    assert 5 == len(gpu.remaining_memory.values())


def test_gpu_check_maximum(gpu):
    with pytest.raises(AssertionError) as exc:
        gpu.can_use_it(0, 25)
    assert 'memory should be between 1 and 24' == str(exc.value)


def test_gpu_check_minimum(gpu):
    with pytest.raises(AssertionError) as exc:
        gpu.can_use_it(0, 0)
    assert 'memory should be between 1 and 24' == str(exc.value)


def test_gpu_check_memory(gpu):
    for gpu_id in range(5):
        if gpu.remaining_memory[gpu_id] > 10000:
            assert gpu.can_use_it(gpu_id, 10)


@pytest.mark.skip(reason="not implemented yet")
def test_gpu_give_me_maximum_gpu(gpu):
    assert False
