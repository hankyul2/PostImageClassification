import copy
import os, sys
import time

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.optim.adam import Adam

sys.path.append('./')

import torch
from easydict import EasyDict as edict
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.gpu import GPU
from src.utils.config import get_config
from src.utils.dataset import POSTDATASET
from src.utils.post_train import get_dataset, make_input
from src.vanila_scafolding import get_model


def get_args() -> edict:
    d = edict({
        'alg': 'supervised',
        'dataset': "cifar10",
        'root': "data",
        'model_name': 'pt2-3',
        'base_model_name': 'cifar10_PL_1620253529',
        'output': "./exp_res",
        'gpu': 4
    })
    return d


class PostTrain:
    def __init__(self):
        args = get_args()
        config = get_config()
        exp_name = self.get_exp_name(args)

        gpu = GPU()
        device = gpu.get_device(args.gpu)

        alg_cfg = config[args.alg]
        dataset_cfg = config[args.dataset]
        base_model_path = os.path.join(args.output, args.base_model_name, 'best_model.pth')
        model, optimizer, ssl_obj = get_model(args, alg_cfg, dataset_cfg, device)
        model.load_state_dict(torch.load(base_model_path))

        post_data_root = os.path.join('data/cifar10', args.base_model_name)
        if not os.path.exists(post_data_root):
            self.make_post_train_data(args, dataset_cfg, model, device, post_data_root)

        l_train_dataset = POSTDATASET(post_data_root, "l_train")
        u_train_dataset = POSTDATASET(post_data_root, "u_train")
        val_dataset = POSTDATASET(post_data_root, "val")
        test_dataset = POSTDATASET(post_data_root, "test")

        self.l_train = DataLoader(l_train_dataset, 100)
        self.u_train = DataLoader(u_train_dataset, 100)
        self.val = DataLoader(val_dataset, 100)
        self.test = DataLoader(test_dataset, 100)

        self.model = model.output
        self.lr = 3e-5
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.device = device
        self.model_save_path = self.get_best_model_save_path(args, exp_name)
        self.model_name = args.model_name

        print("device:", self.device)

    def get_exp_name(self, args) -> str:
        exp_name = ""
        exp_name += str(args.base_model_name) + "_"
        exp_name += str(args.model_name) + "_"
        exp_name += str(int(time.time()))  # unique ID
        return exp_name

    def get_best_model_save_path(self, args, exp_name):
        model_save_path = os.path.join(args.output, exp_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        return model_save_path

    def make_post_train_data(self, args, dataset_cfg, model, device, post_data_root):
        os.makedirs(post_data_root)
        l_train_dataset, u_train_dataset, val_dataset, test_dataset = get_dataset(args, dataset_cfg)
        train_label_path = os.path.join(post_data_root, '{}.npy'.format('l_train'))
        train_unlabel_path = os.path.join(post_data_root, '{}.npy'.format('u_train'))
        test_path = os.path.join(post_data_root, '{}.npy'.format('test'))
        valid_path = os.path.join(post_data_root, '{}.npy'.format('val'))
        make_input(model, l_train_dataset, train_label_path, device)
        make_input(model, val_dataset, train_unlabel_path, device)
        make_input(model, test_dataset, test_path, device)
        make_input(model, val_dataset, valid_path, device)

    @torch.no_grad()
    def test_fn(self, test_set: DataLoader) -> float:
        self.model.eval()

        sum_acc = 0.
        s = time.time()
        for j, data in enumerate(test_set):
            input, target = data
            input, target = input.to(self.device).float(), target.to(self.device).long()

            output = self.model(input)

            pred_label = output.max(1)[1]
            sum_acc += (pred_label == target).float().sum()
            if ((j + 1) % 10) == 0:
                d_p_s = 10 / (time.time() - s)
                print("\r[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                    j + 1, len(test_set), d_p_s, (len(test_set) - j - 1) / d_p_s
                ), end="")
                s = time.time()
        acc = sum_acc / float(len(test_set.dataset))

        self.model.train()
        return acc

    def post_train(self):
        writer = SummaryWriter(os.path.join('run', self.model_name))
        maximum_val_acc = .0

        for iteration in range(100):
            # PseudoLabel = additional_pseudo_label / size_unlabelled_data if additional_pseudo_label != 0 else 0
            # CorrectPseudoLabel = correct_pseudo_label / additional_pseudo_label if correct_pseudo_label != 0 else 0

            self.train_fn2(iteration, writer)
            acc = self.test_fn(self.val)
            print("\n### validation ###")
            print("\nvaridation accuracy : {}".format(acc))

            # test
            if maximum_val_acc < acc:
                maximum_val_acc = acc
                test_acc = self.test_fn(self.test)
                print("\n### test ###")
                print("\ntest accuracy : {}".format(test_acc))

                torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "best_model.pth"))

            writer.add_scalar('Accuracy/Validation', acc * 100, iteration * 100)

        return maximum_val_acc

    def train_fn(self, I, writer, top_k=5, optim=Adam):
        self.model.train()

        for iter, (xs, ys) in enumerate(self.test):
            pseudo_label = 0
            correct_pseudo_label = 0

            loss_org = self.get_loss(self.model, self.l_train)
            x_data = list()
            y_pseudo = list([torch.tensor(-1) for _ in range(100)])

            xs, ys = xs.to(self.device).float(), ys.to(self.device).long()
            output = self.model(xs)
            top_list = torch.topk(output, top_k)[1].long()
            total_size = len(xs)
            for idx, (x, y, y_preds) in enumerate(zip(xs, ys, top_list)):
                min_val = torch.tensor(100.0, device=self.device)
                min_label = torch.tensor(0, device=self.device)
                for y_hat in y_preds:
                    model_copy = copy.deepcopy(self.model)
                    optimizer = optim(model_copy.parameters(), lr=self.lr)
                    y_pred = model_copy(x.reshape(-1, *x.shape))
                    loss = F.cross_entropy(y_pred, y_hat.reshape(-1, *y_hat.shape))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_updated = self.get_loss(model_copy, self.l_train)
                    loss_diff = loss_updated - loss_org

                    if loss_diff < min_val:
                        min_val = loss_diff
                        min_label = y_hat

                if min_val < 0:
                    pseudo_label += 1
                    correct_pseudo_label += int(min_label == y)
                    y_pseudo[idx] = min_label
                    # correct_pseudo_label += int(y_preds[0] == y)
                    # y_pseudo[idx] = y_preds[0]

                # writer.add_scalar('CrossEntropy/Diff', min_val, iter*100 + idx)
                x_data.append(x.reshape(-1, *x.shape))

            y_pseudo = torch.tensor(y_pseudo, device=self.device)
            x_data = torch.cat(x_data)
            y_out = self.model(x_data)
            loss = F.cross_entropy(y_out, y_pseudo.long(), reduction="none", size_average=True, ignore_index=-1).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_val = self.get_loss(self.model, self.val)

            print("[{}] LOSS(train test)=({:3.5f} {:3.5f}) | Pseudo Label={}/{} | Correct Pseudo Label={}/{}".format(iter, loss_org, loss, pseudo_label, 100, correct_pseudo_label, pseudo_label))
            writer.add_scalar('Loss/train', loss_org, I*100 + iter)
            writer.add_scalar('Loss/val', loss_val, I*100 + iter)
            writer.add_scalar('Loss/test', loss, I*100 + iter)
            writer.add_scalar('SSL/PseudoLabelPercent', pseudo_label/total_size*100, I*100 + iter)
            writer.add_scalar('SSL/CorrectPseudoLabelPercent', correct_pseudo_label/pseudo_label*100, I*100 + iter)
        print(
            "===========================================================================================================")

    def train_fn2(self, I, writer, top_k=5, optim=Adam):
        self.model.train()
        for iter, (xs, ys) in enumerate(self.test):
            pseudo_label = 0
            correct_pseudo_label = 0

            loss_org = self.get_loss(self.model, self.l_train)
            y_pseudo = list([torch.tensor(-1) for _ in range(100)])

            xs, ys = xs.to(self.device).float(), ys.to(self.device).long()
            output = self.model(xs)
            top_prob, top_list = torch.topk(output, top_k)
            total_size = len(xs)
            for idx, (x, y, y_probs, y_preds) in enumerate(zip(xs, ys, top_prob, top_list)):
                if y_probs[0] > 0.95:
                    pseudo_label += 1
                    correct_pseudo_label += int(y_preds[0] == y)
                    y_pseudo[idx] = y_preds[0]

            y_pseudo = torch.tensor(y_pseudo, device=self.device)
            y_out = self.model(xs)
            loss = F.cross_entropy(y_out, y_pseudo.long(), reduction="none", size_average=True, ignore_index=-1).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_val = self.get_loss(self.model, self.val)

            print("[{}] LOSS(train test)=({:3.5f} {:3.5f}) | Pseudo Label={}/{} | Correct Pseudo Label={}/{}".format(iter, loss_org, loss, pseudo_label, total_size, correct_pseudo_label, pseudo_label))
            writer.add_scalar('Loss/train', loss_org, I*100 + iter)
            writer.add_scalar('Loss/val', loss_val, I*100 + iter)
            writer.add_scalar('Loss/test', loss, I*100 + iter)
            writer.add_scalar('SSL/PseudoLabelPercent', pseudo_label/total_size*100, I*100 + iter)
            writer.add_scalar('SSL/CorrectPseudoLabelPercent', correct_pseudo_label/pseudo_label*100, I*100 + iter)
        print(
            '===========================================================================================================')


    def get_loss(self, model, train):
        loss = torch.tensor(0.0, device=self.device)
        for x, y in train:
            x, y = x.to(self.device).float(), y.to(self.device).long()
            output = model(x)
            loss += F.cross_entropy(output, y).mean()
        return loss

if __name__ == '__main__':
    tool = PostTrain()
    tool.post_train()