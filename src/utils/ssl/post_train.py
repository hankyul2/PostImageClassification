import copy

import torch
from torch import nn
import torch.nn.functional as F


class PT(nn.Module):
    def __init__(self, top_k, optim, lr):
        super().__init__()
        self.lr = lr
        self.optim = optim
        self.top_k = top_k
        self.th = 0.95


    def forward(self, x, y, model, mask, real_target, maximum_val_acc):
        pseudo_label = 0
        correct_pseudo_label = 0

        split_idx = len(real_target) // 2
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
                    optimizer = self.optim(model_copy.parameters(), lr=self.lr)

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
                    pseudo_label += 1
                    correct_pseudo_label += int(min_label == y_ans)
                    gt_mask[idx] = 1
                    y_pseudo[idx] = min_label

        loss = F.cross_entropy(outputs, y_pseudo.long(), reduction="none", ignore_index=-1).mean()

        return loss, pseudo_label, correct_pseudo_label

    def __make_one_hot(self, y, n_classes=10):
        return torch.eye(n_classes)[y].to(y.device)
