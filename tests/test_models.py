import GPUtil
import torch
from easydict import EasyDict
from torch import nn

from src.postTrain.models import get_network
from src.postTrain.resnet import ResNet
from src.postTrain.wresnet import conv3x3, relu


def test_get_resnet_network():
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


def test_get_wresnet_model():
    pass


def test_construction_conv3x3():
    conv = conv3x3(2, 2)
    assert conv.weight.shape == (2, 2, 3, 3)
    assert conv.bias == None


def test_function_conv3x3():
    conv = conv3x3(3, 2)
    a = torch.rand((100, 3, 32, 32))
    b = conv(a)
    assert b.shape == (100, 2, 32, 32)

def test_relu_construction():
    activation = relu()
    assert isinstance(activation, nn.LeakyReLU)

def test_relu_function():
    activation = relu()
    a = torch.tensor([-1.0])
    assert activation(a) == torch.tensor([-0.1])
