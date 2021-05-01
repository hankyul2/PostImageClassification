from src.postTrain.resnet import ResNet, BasicBlock, BottleNeck


def get_network(args):
    if args.net == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2])
    elif args.net == "resnet34":
        return ResNet(BasicBlock, [3, 4, 6, 3])
    elif args.net == "resnet50":
        return ResNet(BottleNeck, [3, 4, 6, 3])
    elif args.net == "resnet101":
        return ResNet(BottleNeck, [3, 4, 23, 3])
    elif args.net == "resnet152":
        return ResNet(BottleNeck, [3, 8, 36, 3])

