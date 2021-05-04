from src.gpu import GPU
from src.monitor import Monitor

from easydict import EasyDict as edict

from src.postTrain.models import get_network


def main(args):
    gpu = GPU()
    gpu.show_gpu_memory()
    gpu.can_use_it(args.gpu, args.memory)
    monitor = Monitor(args.print_freq, args.gpu)
    net = get_network(args)
    print("hello")
    monitor.stop()

if "__main__" == __name__:
    args = edict({
        "net":"resnet50",
        "gpu":3,
        "memory":24,
        "numb_worker":4,
        "print_freq":60,
        "b":128,
        "warm":1,
        "lr":0.1,
        "resume":False
    })
    main(args)