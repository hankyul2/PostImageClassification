import math, time, json, os, sys
from easydict import EasyDict as edict

sys.path.append('./')


import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.gpu import GPU
from src.postTrain.wresnet import WRN
from src.utils.config import get_config
from src.utils.dataset import transform, RandomSampler
from src.utils.ssl.post_train import PT
from src.utils.ssl.pseudo_label import PL
from src.utils.ssl.mixmatch import MixMatch
from src.utils.data import download_preprocess_save_data


def get_args() -> edict:
    d = edict({
        'alg': "MM",
        'em': 0,
        'validation': 820,
        'dataset': "cifar10",
        'root': "data",
        'model_name':'mixmatch1',
        'output': "./exp_res",
        'gpu':4
    })
    return d

def main():
    condition = {}
    exp_name = ""
    args = get_args()
    config = get_config()

    exp_name += str(args.dataset) + "_"
    exp_name += str(args.alg) + "_"
    exp_name += str(int(time.time()))  # unique ID
    if not os.path.exists(os.path.join(args.output, exp_name)):
        os.makedirs(os.path.join(args.output, exp_name))

    dataset_cfg = config[args.dataset]
    shared_cfg = config["shared"]
    alg_cfg = config[args.alg]

    condition["dataset"] = args.dataset
    condition["algorithm"] = args.alg
    condition["h_parameters"] = alg_cfg

    print("dataset : {}".format(args.dataset))
    print("algorithm : {}".format(args.alg))
    print("parameters : ", alg_cfg)

    gpu = GPU()
    device = gpu.get_device(args.gpu)
    download_preprocess_save_data()
    l_loader, u_loader, val_loader, test_loader, val_dataset, test_dataset = get_data(args, dataset_cfg, shared_cfg)
    model, optimizer, ssl_obj = get_model(args, alg_cfg, dataset_cfg, device)

    print()
    iteration = 0
    maximum_val_acc = 0
    s = time.time()

    writer = SummaryWriter(os.path.join('run', args.model_name))

    size_unlabelled_data = shared_cfg['batch_size'] // 2 * args.validation
    additional_pseudo_label = 0
    correct_pseudo_label = 0

    for l_data, u_data in zip(l_loader, u_loader):
        iteration += 1
        l_input, target = l_data
        l_input, target = l_input.to(device).float(), target.to(device).long()

        if args.alg != "supervised":  # for ssl algorithm
            u_input, dummy_target, real_target = u_data
            u_input, dummy_target, real_target = u_input.to(device).float(), dummy_target.to(
                device).long(), real_target.to(device).long()

            target = torch.cat([target, dummy_target], 0)
            unlabeled_mask = (target == -1).float()

            inputs = torch.cat([l_input, u_input], 0)
            outputs = model(inputs)

            # ramp up exp(-5(1 - t)^2)
            coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration / shared_cfg["warmup"], 1)) ** 2)
            ssl_loss, additional_pseudo_label_iter, correct_pseudo_label_iter = ssl_obj(inputs, outputs.detach(), model,
                                                                                        unlabeled_mask, real_target)
            additional_pseudo_label += additional_pseudo_label_iter
            correct_pseudo_label += correct_pseudo_label_iter
            ssl_loss *= coef
        else:
            outputs = model(l_input)
            coef = 0
            ssl_loss = torch.zeros(1).to(device)
        cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()
        loss = cls_loss + ssl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration == 1 or (iteration % 100) == 0:
            wasted_time = time.time() - s
            rest = (shared_cfg["iteration"] - iteration) / 100 * wasted_time / 60
            print(
                "\riteration [{}/{}] cls loss : {:.6e}, SSL loss : {:.6e}, coef : {:.5e}, time : {:.3f} iter/sec, rest : {:.3f} min, lr : {}".format(
                    iteration, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), coef, 100 / wasted_time, rest,
                    optimizer.param_groups[0]["lr"]), end="")
            s = time.time()

        # validation
        if (iteration % args.validation) == 0 or iteration == shared_cfg["iteration"]:
            with torch.no_grad():
                model.eval()
                print()
                print("### validation ###")
                sum_acc = 0.
                s = time.time()
                for j, data in enumerate(val_loader):
                    input, target = data
                    input, target = input.to(device).float(), target.to(device).long()

                    output = model(input)

                    pred_label = output.max(1)[1]
                    sum_acc += (pred_label == target).float().sum()
                    if ((j + 1) % 10) == 0:
                        d_p_s = 10 / (time.time() - s)
                        print("\r[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                            j + 1, len(val_loader), d_p_s, (len(val_loader) - j - 1) / d_p_s
                        ), end="")
                        s = time.time()
                acc = sum_acc / float(len(val_dataset))
                print()
                print("varidation accuracy : {}".format(acc))
                PseudoLabel = additional_pseudo_label / size_unlabelled_data if additional_pseudo_label != 0 else 0
                CorrectPseudoLabel = correct_pseudo_label / additional_pseudo_label if correct_pseudo_label != 0 else 0
                writer.add_scalar('Accuracy/Validation', acc * 100, iteration)
                writer.add_scalar('SSL/PseudoLabelPercent', PseudoLabel * 100, iteration)
                writer.add_scalar('SSL/CorrectPseudoLabelPercent', CorrectPseudoLabel * 100, iteration)
                additional_pseudo_label = correct_pseudo_label = 0
                # test
                if maximum_val_acc < acc:
                    print("### test ###")
                    maximum_val_acc = acc
                    sum_acc = 0.
                    s = time.time()
                    for j, data in enumerate(test_loader):
                        input, target = data
                        input, target = input.to(device).float(), target.to(device).long()
                        output = model(input)
                        pred_label = output.max(1)[1]
                        sum_acc += (pred_label == target).float().sum()
                        if ((j + 1) % 10) == 0:
                            d_p_s = 100 / (time.time() - s)
                            print("\r[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                                j + 1, len(test_loader), d_p_s, (len(test_loader) - j - 1) / d_p_s
                            ), end="")
                            s = time.time()
                    print()
                    test_acc = sum_acc / float(len(test_dataset))
                    print("test accuracy : {}".format(test_acc))
                    torch.save(model.state_dict(), os.path.join(args.output, exp_name, "best_model.pth"))
            model.train()
            s = time.time()
        # lr decay
        if iteration == shared_cfg["lr_decay_iter"]:
            optimizer.param_groups[0]["lr"] *= shared_cfg["lr_decay_factor"]

    print("test acc : {}".format(test_acc))
    condition["test_acc"] = test_acc.item()

    with open(os.path.join(args.output, exp_name, "final_model.json"), "w") as f:
        json.dump(condition, f)

def get_data(args, dataset_cfg, shared_cfg):
    l_train_dataset = dataset_cfg["dataset"](args.root, "l_train")
    u_train_dataset = dataset_cfg["dataset"](args.root, "u_train")
    val_dataset = dataset_cfg["dataset"](args.root, "val")
    test_dataset = dataset_cfg["dataset"](args.root, "test")

    print("labeled data : {}, unlabeled data : {}, training data : {}".format(
        len(l_train_dataset), len(u_train_dataset), len(l_train_dataset) + len(u_train_dataset)))

    if args.alg != "supervised":
        # batch size = 0.5 x batch size
        l_loader = DataLoader(
            l_train_dataset, shared_cfg["batch_size"] // 2, drop_last=True,
            sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"] // 2)
        )
    else:
        l_loader = DataLoader(
            l_train_dataset, shared_cfg["batch_size"], drop_last=True,
            sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"])
        )

    u_loader = DataLoader(
        u_train_dataset, shared_cfg["batch_size"] // 2, drop_last=True,
        sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"] // 2)
    )

    val_loader = DataLoader(val_dataset, 128, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)

    print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

    return l_loader, u_loader, val_loader, test_loader, val_dataset, test_dataset


def get_model(args, alg_cfg, dataset_cfg, device):
    transform_fn = transform(*dataset_cfg["transform"])
    model = WRN(4, dataset_cfg["num_classes"], transform_fn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])

    trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
    print("trainable parameters : {}".format(trainable_paramters))

    if args.alg == "PL":  # pseudo label
        ssl_obj = PL(alg_cfg["threashold"])
    elif args.alg == "PT":  # post train
        ssl_obj = PT(alg_cfg["top_k"], optim.Adam, alg_cfg['lr'])
    elif args.alg == "MM":  # MixMatch
        ssl_obj = MixMatch(alg_cfg["T"], alg_cfg["K"], alg_cfg["alpha"])
    elif args.alg == "supervised":
        ssl_obj = None
    else:
        raise ValueError("{} is unknown algorithm".format(args.alg))

    return model, optimizer, ssl_obj

if __name__ == '__main__':
    main()

