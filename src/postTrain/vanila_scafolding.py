import os,math, time, json, os, copy
os.chdir('/content/drive/Shareddrives/ColabForever/SSL')
import numpy as np
import bsconv.pytorch
from easydict import EasyDict as edict
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter



def split_l_u(train_set, n_labels):
    # NOTE: this function assume that train_set is shuffled.
    images, labels = train_set.values()
    classes = np.unique(labels)
    n_labels_per_cls = n_labels // len(classes)
    l_images, l_labels, u_images, u_labels, u_real_labels = [[] for _ in range(5)]
    for c in classes:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:]]
        u_real_labels += [c_labels[n_labels_per_cls:]]
        u_labels += [np.zeros_like(c_labels[n_labels_per_cls:]) - 1] # dammy label
    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0), "real_labels": np.concatenate(u_real_labels, 0)}
    return l_train_set, u_train_set


def split_validation(train_set, validation_count):
    train_images = train_set["images"][validation_count:]
    train_labels = train_set["labels"][validation_count:]
    validation_images = train_set["images"][:validation_count]
    validation_labels = train_set["labels"][:validation_count]
    validation_set = {"images": validation_images, "labels": validation_labels}
    train_set = {"images": train_images, "labels": train_labels}
    return train_set, validation_set


def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)


def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp


def gcn(images, multiplier=55, eps=1e-10):
    # global contrast normalization
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    return multiplier * images / per_image_norm


def load_and_gcn(data_dir):
    splits = []
    for train in [True, False]:
        data_org = datasets.CIFAR10(data_dir, train, download=True)
        data_new = {"images":gcn(data_org.data), "labels":np.array(data_org.targets)}
        splits.append(data_new)
    return splits


def download_preprocess_save_data(seed=1, dataset='cifar10', nlabels=4000, data_dir='./data'):
    if os.path.exists(os.path.join(data_dir, dataset, 'test.npy')):
        return
    COUNTS = {"cifar10": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0}}
    rng = np.random.RandomState(seed)
    validation_count = COUNTS[dataset]['valid']
    if dataset == 'cifar10':
        train_set, test_set = load_and_gcn(data_dir)
        mean, zca_decomp = get_zca_normalization_param(train_set["images"])
        train_set["images"] = zca_normalization(train_set["images"], mean, zca_decomp)
        test_set["images"] = zca_normalization(test_set["images"], mean, zca_decomp)
        train_set["images"] = np.transpose(train_set["images"], (0,3,1,2))
        test_set["images"] = np.transpose(test_set["images"], (0,3,1,2))
    indices = rng.permutation(len(train_set['images']))
    train_set['images'] = train_set['images'][indices]
    train_set['labels'] = train_set['labels'][indices]
    train_set, validation_set = split_validation(train_set, validation_count)
    l_train_set, u_train_set = split_l_u(train_set, nlabels)
    if not os.path.exists(os.path.join(data_dir, dataset)):
        os.mkdir(os.path.join(data_dir, dataset))
    datasets = dict(l_train=l_train_set, u_train=u_train_set, val=validation_set, test=test_set)
    for key, value in datasets.items():
        np.save(os.path.join(data_dir, dataset, key), value)


def get_config():
    class CIFAR10:
        def __init__(self, root, split="l_train"):
            self.split=split
            self.dataset = np.load(os.path.join(root, "cifar10", split+".npy"), allow_pickle=True).item()

        def __getitem__(self, idx):
            image = self.dataset["images"][idx]
            label = self.dataset["labels"][idx]
            if self.split=='u_train':
                real_label = self.dataset["real_labels"][idx]
                return image, label, real_label
            return image, label

        def __len__(self):
            return len(self.dataset["images"])
    shared_config = {
        "iteration" : 500000,
        "warmup" : 200000,
        "lr_decay_iter" : 400000,
        "lr_decay_factor" : 0.2,
        "batch_size" : 100,
    }
    cifar10_config = {
        "transform" : [True, True, True],
        "dataset" : CIFAR10,
        "num_classes" : 10,
    }
    pl_config = {
        # pseudo label
        "threashold" : 0.95,
        "lr" : 3e-4,
        "consis_coef" : 1,
    }
    pt_config = {
        # post train
        "top_k":5,
        "lr" : 3e-4,
        "consis_coef" : 1,
    }
    supervised_config = {
        "lr" : 3e-3
    }
    config = {
        "shared" : shared_config,
        "cifar10" : cifar10_config,
        "supervised" : supervised_config,
        "PL" : pl_config,
        "PT" : pt_config,
    }
    return config


def get_args():
    d = edict({
        'alg':"PT",
        'em':0,
        'validation':820,
        'dataset':"cifar10",
        'root':"data",
        'output':"./exp_res"
        })
    return d


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    return device


def transform_fn(x, flip=True, r_crop=True, g_noise=True):
    if self.flip and random.random() > 0.5:
        x = x.flip()

def get_data(args):
    import random
    
    class transform:
        def __init__(self, flip=True, r_crop=True, g_noise=True):
            self.flip = flip
            self.r_crop = r_crop
            self.g_noise = g_noise
            print("holizontal flip : {}, random crop : {}, gaussian noise : {}".format(
                self.flip, self.r_crop, self.g_noise
            ))

        def __call__(self, x):
            if self.flip and random.random() > 0.5:
                x = x.flip(-1)
            if self.r_crop:
                h, w = x.shape[-2:]
                x = F.pad(x, [2,2,2,2], mode="reflect")
                l, t = random.randint(0, 4), random.randint(0,4)
                x = x[:,:,t:t+h,l:l+w]
            if self.g_noise:
                n = torch.randn_like(x) * 0.15
                x = n + x
            return x

    class RandomSampler(torch.utils.data.Sampler):
        """ sampling without replacement """
        def __init__(self, num_data, num_sample):
            iterations = num_sample // num_data + 1
            self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    condition = {}
    exp_name = ""

    print("dataset : {}".format(args.dataset))
    condition["dataset"] = args.dataset
    exp_name += str(args.dataset) + "_"
    dataset_cfg = config[args.dataset]
    transform_fn = transform(*dataset_cfg["transform"])

    l_train_dataset = dataset_cfg["dataset"](args.root, "l_train")
    u_train_dataset = dataset_cfg["dataset"](args.root, "u_train")
    val_dataset = dataset_cfg["dataset"](args.root, "val")
    test_dataset = dataset_cfg["dataset"](args.root, "test")

    print("labeled data : {}, unlabeled data : {}, training data : {}".format(
    len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))

    shared_cfg = config["shared"]
    if args.alg != "supervised":
        # batch size = 0.5 x batch size
        l_loader = DataLoader(
            l_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
            sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
        )
    else:
        l_loader = DataLoader(
            l_train_dataset, shared_cfg["batch_size"], drop_last=True,
            sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"])
        )
    
    print("algorithm : {}".format(args.alg))
    condition["algorithm"] = args.alg
    exp_name += str(args.alg) + "_"

    u_loader = DataLoader(
        u_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
        sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
    )

    val_loader = DataLoader(val_dataset, 128, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)

    print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

    return exp_name, condition, dataset_cfg, transform_fn, shared_cfg, l_loader, u_loader, val_loader, test_loader, val_dataset, test_dataset


def get_model(device, config, args, condition, dataset_cfg, transform_fn):
    def conv3x3(i_c, o_c, stride=1):
        return nn.Conv2d(i_c, o_c, 3, stride, 1, bias=False)

    class BatchNorm2d(nn.BatchNorm2d):
        def __init__(self, channels, momentum=1e-3, eps=1e-3):
            super().__init__(channels)
            self.update_batch_stats = True

        def forward(self, x):
            if self.update_batch_stats:
                return super().forward(x)
            else:
                return nn.functional.batch_norm(
                    x, None, None, self.weight, self.bias, True, self.momentum, self.eps
                )

    def relu():
        return nn.LeakyReLU(0.1)

    class residual(nn.Module):
        def __init__(self, input_channels, output_channels, stride=1, activate_before_residual=False):
            super().__init__()
            layer = []
            if activate_before_residual:
                self.pre_act = nn.Sequential(
                    BatchNorm2d(input_channels),
                    relu()
                )
            else:
                self.pre_act = nn.Identity()
                layer.append(BatchNorm2d(input_channels))
                layer.append(relu())
            layer.append(conv3x3(input_channels, output_channels, stride))
            layer.append(BatchNorm2d(output_channels))
            layer.append(relu())
            layer.append(conv3x3(output_channels, output_channels))

            if stride >= 2 or input_channels != output_channels:
                self.identity = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
            else:
                self.identity = nn.Identity()

            self.layer = nn.Sequential(*layer)

        def forward(self, x):
            x = self.pre_act(x)
            return self.identity(x) + self.layer(x)

    class WRN(nn.Module):
        """ WRN28-width with leaky relu (negative slope is 0.1)"""
        def __init__(self, width, num_classes, transform_fn=None):
            super().__init__()

            self.init_conv = conv3x3(3, 16)

            filters = [16, 16*width, 32*width, 64*width]

            unit1 = [residual(filters[0], filters[1], activate_before_residual=True)] + \
                [residual(filters[1], filters[1]) for _ in range(1, 4)]
            self.unit1 = nn.Sequential(*unit1)

            unit2 = [residual(filters[1], filters[2], 2)] + \
                [residual(filters[2], filters[2]) for _ in range(1, 4)]
            self.unit2 = nn.Sequential(*unit2)

            unit3 = [residual(filters[2], filters[3], 2)] + \
                [residual(filters[3], filters[3]) for _ in range(1, 4)]
            self.unit3 = nn.Sequential(*unit3)

            self.unit4 = nn.Sequential(*[BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)])

            self.output = nn.Linear(filters[3], num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")    
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

            self.transform_fn = transform_fn

        def forward(self, x, return_feature=False):
            if self.training and self.transform_fn is not None:
                x = self.transform_fn(x)
            x = self.init_conv(x)
            x = self.unit1(x)
            x = self.unit2(x)
            x = self.unit3(x)
            f = self.unit4(x)
            c = self.output(f.squeeze())
            if return_feature:
                return [c, f]
            else:
                return c

        def update_batch_stats(self, flag):
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.update_batch_stats = flag
    alg_cfg = config[args.alg]
    print("parameters : ", alg_cfg)
    condition["h_parameters"] = alg_cfg

    replacer = bsconv.pytorch.BSConvS_Replacer()
    model = WRN(4, dataset_cfg["num_classes"], transform_fn)
    model = replacer.apply(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])

    trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
    print("trainable parameters : {}".format(trainable_paramters))

    if args.alg == "PL": # pseudo label
        class PL(nn.Module):
            def __init__(self, threshold):
                super().__init__()
                self.th = threshold

            def forward(self, x, y, model, mask, real_target):
                y_probs = y.softmax(1)
                onehot_label = self.__make_one_hot(y_probs.max(1)[1]).float()
                gt_mask = (y_probs > self.th).float()
                gt_mask = gt_mask.max(1)[0] # reduce_any
                lt_mask = 1 - gt_mask # logical not
                p_target = gt_mask[:,None] * 10 * onehot_label + lt_mask[:,None] * y_probs
                model.update_batch_stats(False)
                output = model(x)
                loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1)*mask).mean()
                model.update_batch_stats(True)

                pseudo_mask = mask.bool() & gt_mask.bool()
                y_pred = y_probs.max(1)[1]
                pseudo_selected = torch.masked_select(y_pred, pseudo_mask)
                real_selected = torch.masked_select(torch.cat([torch.zeros_like(real_target), real_target], 0), pseudo_mask)
                pseudo_label = len(pseudo_selected)
                correct_pseudo_label = (pseudo_selected==real_selected).sum().item()
                # print("{}/{}".format(correct_pseudo_label,pseudo_label))

                return loss, pseudo_label, correct_pseudo_label

            def __make_one_hot(self, y, n_classes=10):
                return torch.eye(n_classes)[y].to(y.device)
        ssl_obj = PL(alg_cfg["threashold"])
    elif args.alg == "PT": # post train
        class PT(nn.Module):
            def __init__(self, top_k):
                super().__init__()
                self.top_k = top_k
                self.th = 0.95

            def forward(self, x, y, model, mask, real_target, maximum_val_acc):
                pseudo_label = 0
                correct_pseudo_label = 0
                
                split_idx = len(real_target)// 2
                x_label, x_unlabelled = x[:split_idx], x[split_idx:]
                y_label, y_unlabelled = real_target[:split_idx], real_target[split_idx:]
                
                model.update_batch_stats(False)
                outputs = model(x)
                y_pred_label, _ = outputs[:split_idx], outputs[split_idx:]
                cls_loss_org = F.cross_entropy(y_pred_label, y_label).mean()
                model.update_batch_stats(True)

                top_k = torch.topk(y, self.top_k)[1].long()

                gt_mask = torch.zeros_like(mask)
                y_pseudo = torch.zeros_like(mask) - 1

                if maximum_val_acc > 30:
                    for idx, (x, tops, y_ans) in enumerate(zip(x, top_k, real_target)):
                        if idx < split_idx: 
                            continue
                        
                        min_val = 1000
                        min_label = 0

                        for y_hat in tops:
                            model_copy = copy.deepcopy(model)
                            optimizer = optim.Adam(model_copy.parameters(), lr=alg_cfg["lr"])

                            model_copy.update_batch_stats(False)
                            y_pred = model_copy(x.reshape(-1, *x.shape))
                            loss = F.cross_entropy(y_pred.reshape(-1, *y_pred.shape), y_hat.reshape(-1, *y_hat.shape))
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                            y_pred_label = model_copy(x_label)
                            cls_loss = F.cross_entropy(y_pred_label, y_label).mean()
                            cls_loss -= cls_loss_org

                            if cls_loss < min_val:
                                min_val = cls_loss
                                min_label = y_hat
                        
                        if min_val < 0:
                            pseudo_label+=1
                            correct_pseudo_label += int(min_label==y_ans)
                            gt_mask[idx] = 1
                            y_pseudo[idx] = min_label
                
                loss = F.cross_entropy(outputs, y_pseudo.long(), reduction="none", ignore_index=-1).mean()

                return loss, pseudo_label, correct_pseudo_label

            def __make_one_hot(self, y, n_classes=10):
                return torch.eye(n_classes)[y].to(y.device)
        ssl_obj = PT(alg_cfg["top_k"])
    elif args.alg == "supervised":
        ssl_obj = None
    else:
        raise ValueError("{} is unknown algorithm".format(args.alg))

    return model, optimizer, alg_cfg, ssl_obj


download_preprocess_save_data()
config = get_config()
args = get_args()
device = get_device()
exp_name, condition, dataset_cfg, transform_fn, shared_cfg, l_loader, u_loader, val_loader, test_loader, val_dataset, test_dataset = get_data(args)
model, optimizer, alg_cfg, ssl_obj = get_model(device, config, args, condition, dataset_cfg, transform_fn)

print()
iteration = 0
maximum_val_acc = 0
s = time.time()

writer = SummaryWriter('runs/pt')

size_unlabelled_data = shared_cfg['batch_size'] // 2 * args.validation 
additional_pseudo_label = 0
correct_pseudo_label = 0 

for l_data, u_data in zip(l_loader, u_loader):
    iteration += 1
    l_input, target = l_data
    l_input, target = l_input.to(device).float(), target.to(device).long()
    
    
    if args.alg != "supervised": # for ssl algorithm
        u_input, dummy_target, real_target = u_data
        u_input, dummy_target, real_target = u_input.to(device).float(), dummy_target.to(device).long(), real_target.to(device).long()
        # changes 1
        real_target = torch.cat([target, real_target], 0)
        target = torch.cat([target, dummy_target], 0)
        unlabeled_mask = (target == -1).float()

        inputs = torch.cat([l_input, u_input], 0)
        outputs = model(inputs)

        # ramp up exp(-5(1 - t)^2)
        coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration/shared_cfg["warmup"], 1))**2)
        # changes 2
        ssl_loss, additional_pseudo_label_iter, correct_pseudo_label_iter = ssl_obj(inputs, outputs.detach(), model, unlabeled_mask, real_target, maximum_val_acc)
        additional_pseudo_label += additional_pseudo_label_iter
        correct_pseudo_label += correct_pseudo_label_iter
        ssl_loss *= coef
    else:
        outputs = model(l_input)
        coef = 0
        ssl_loss = torch.zeros(1).to(device)
    cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()
    loss = cls_loss + ssl_loss + model.reg_loss(alpha=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iteration == 1 or (iteration % 100) == 0:
        wasted_time = time.time() - s
        rest = (shared_cfg["iteration"] - iteration)/100 * wasted_time / 60
        print("\riteration [{}/{}] cls loss : {:.6e}, SSL loss : {:.6e}, coef : {:.5e}, time : {:.3f} iter/sec, rest : {:.3f} min, lr : {}".format(
            iteration, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), coef, 100 / wasted_time, rest, optimizer.param_groups[0]["lr"]), end="")
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
                if ((j+1) % 10) == 0:
                    d_p_s = 10/(time.time()-s)
                    print("\r[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                        j+1, len(val_loader), d_p_s, (len(val_loader) - j-1)/d_p_s
                    ), end="")
                    s = time.time()
            acc = sum_acc/float(len(val_dataset))
            print()
            print("varidation accuracy : {}".format(acc))
            PseudoLabel = additional_pseudo_label / size_unlabelled_data if additional_pseudo_label != 0 else 0
            CorrectPseudoLabel = correct_pseudo_label / additional_pseudo_label if correct_pseudo_label != 0 else 0
            writer.add_scalar('Accuracy/Validation', acc*100, iteration)
            writer.add_scalar('SSL/PseudoLabelPercent', PseudoLabel*100, iteration)
            writer.add_scalar('SSL/CorrectPseudoLabelPercent', CorrectPseudoLabel*100, iteration)
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
                    if ((j+1) % 10) == 0:
                        d_p_s = 100/(time.time()-s)
                        print("\r[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                            j+1, len(test_loader), d_p_s, (len(test_loader) - j-1)/d_p_s
                        ), end="")
                        s = time.time()
                print()
                test_acc = sum_acc / float(len(test_dataset))
                print("test accuracy : {}".format(test_acc))
                # torch.save(model.state_dict(), os.path.join(args.output, "best_model.pth"))
        model.train()
        s = time.time()
    # lr decay
    if iteration == shared_cfg["lr_decay_iter"]:
        optimizer.param_groups[0]["lr"] *= shared_cfg["lr_decay_factor"]

print("test acc : {}".format(test_acc))
condition["test_acc"] = test_acc.item()

exp_name += str(int(time.time())) # unique ID
if not os.path.exists(args.output):
    os.mkdir(args.output)
with open(os.path.join(args.output, exp_name + ".json"), "w") as f:
    json.dump(condition, f)
# d_pseudo = list()

# while True:
#     train(a, d)
#     loss_prev, out_softmax = loss_fn(predict(a, d.x), d.y), predict(a, d_unlabelled.x)

#     for x_idx, out_row in enumerate(out_softmax):
#         x = d_unlabelled.x[x_idx]
#         top_5, top_5_idx = topk(out_row, 5)
#         losses = []
#         for label_candidate in top_5_idx:
#             a_copy = copy_model(a, get_model)
#             train(a_copy, {'data': x, 'label': label_candidate})
#             out_softmax = predict(a, d.x)
#             loss_now = loss_fn(out_softmax, d_unlabelled.y)  # cross_entropy or mse_loss or BCE or AUC
#             loss_post = loss_now - loss_prev  # this part is depend on loss_fn
#             losses.append(loss_post)
#         lowest_entropy, index = losses.min(dim=0)
#         if lowest_entropy < 0:
#             d_unlabelled.remove(top_5_idx[index[0]])
#             d_pseudo.append({'data': x, 'label': top_5_idx[index[0]]})

#     d = combine_data([d, d_pseudo])
#     d_pseudo = list()
