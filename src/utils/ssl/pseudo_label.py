import torch
from torch import nn
import torch.nn.functional as F

class PL(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.th = threshold

    def forward(self, x, y, model, mask, real_target):
        y_probs = y.softmax(1)
        onehot_label = self.__make_one_hot(y_probs.max(1)[1]).float()
        gt_mask = (y_probs > self.th).float()
        gt_mask = gt_mask.max(1)[0]  # reduce_any
        lt_mask = 1 - gt_mask  # logical not
        p_target = gt_mask[:, None] * 10 * onehot_label + lt_mask[:, None] * y_probs
        model.update_batch_stats(False)
        output = model(x)
        loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1) * mask).mean()
        model.update_batch_stats(True)

        pseudo_mask = mask.bool() & gt_mask.bool()
        y_pred = y_probs.max(1)[1]
        pseudo_selected = torch.masked_select(y_pred, pseudo_mask)
        real_selected = torch.masked_select(torch.cat([torch.zeros_like(real_target), real_target], 0),
                                            pseudo_mask)
        pseudo_label = len(pseudo_selected)
        correct_pseudo_label = (pseudo_selected == real_selected).sum().item()
        # print("{}/{}".format(correct_pseudo_label,pseudo_label))

        return loss, pseudo_label, correct_pseudo_label

    def __make_one_hot(self, y, n_classes=10):
        return torch.eye(n_classes)[y].to(y.device)