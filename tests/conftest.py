import pytest

from src.gpu import GPU


@pytest.fixture
def gpu() -> GPU:
    g = GPU()
    return g