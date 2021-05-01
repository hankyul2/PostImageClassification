import GPUtil
from easydict import EasyDict

from src.postTrain.models import get_network
from src.postTrain.resnet import ResNet


def test_get_network():
    args = EasyDict({
        "net":"resnet50",
        "gpu":GPUtil.getFirstAvailable(),
        "memory":24,
        "numb_worker":4,
        "print_freq":60,
        "b":128,
        "warm":1,
        "lr":0.1,
        "resume":False
    })
    assert isinstance(get_network(args), ResNet)


def test_get_model():
    assert False
