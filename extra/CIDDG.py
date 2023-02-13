import os
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradMultiplyLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, hidden_layer=1024, num_source = 6):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, num_source)
        self.activation = nn.LogSoftmax(dim=1)
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.activation(x)
        return x





class CIDDG(nn.Module):
    def __init__(self, alpha, num_classes):
        """
        Input:
            E: encoder
            M: classifier
            D: discriminator
            alpha: weighting parameter of label classifier and domain classifier
            num_classes: the number of classes
         """
        super(CIDDG, self).__init__()
        self.D = Discriminator()
        self.Ds = [deepcopy(self.D).cuda() for _ in range(num_classes)]
        self._alpha = alpha
        self._current_step = None
        self._n_steps = None
        self._annealing_func = None
        self._lr_scheduler = None

    def forward(self, feature, use_reverse_layer=True):
        domain_outputs = []
        for D in self.Ds:
            if use_reverse_layer:
                domain_output = D(ReverseLayerF.apply(feature, self._current_alpha))
            else:
                domain_output = D(GradMultiplyLayerF.apply(feature, self._current_alpha))
            domain_outputs.append(domain_output)
        return domain_outputs

    def pred_d_by_D(self, feature, use_reverse_layer=True):
        if use_reverse_layer:
            domain_output = self.D(ReverseLayerF.apply(feature, self._current_alpha))
        else:
            domain_output = self.D(GradMultiplyLayerF.apply(feature, self._current_alpha))
        return domain_output

    def set_alpha_scheduler(self, n_steps, annealing_func='exp'):
        self._current_step = 0
        self._n_steps = n_steps
        self._annealing_func = annealing_func

    def alpha_scheduler_step(self):
        self._current_step += 1

    def set_lr_scheduler(self, optimizer, n_steps, lr0, lamb=None):
        if lamb is None:
            lamb = lambda current_step: lr0 / ((1 + 10 * (current_step / n_steps)) ** 0.75)
        scheduler = LambdaLR(optimizer, lr_lambda=[lamb])
        self._lr_scheduler = scheduler

    def lr_scheduler_step(self):
        self._lr_scheduler.step()

    def scheduler_step(self):
        if self._annealing_func is not None:
            self.alpha_scheduler_step()
        if self._lr_scheduler is not None:
            self.lr_scheduler_step()

    # @property
    # def _current_alpha(self):
    #     if self._annealing_func is None:
    #         return self._alpha
    #     elif self._annealing_func == 'exp':
    #         p = float(self._current_step) / self._n_steps
    #         return float((2. / (1. + np.exp(-10 * p)) - 1) * self._alpha)
    #     else:
    #         raise Exception()

    def set_current_alpha(self,alpha):
        self._current_alpha = alpha