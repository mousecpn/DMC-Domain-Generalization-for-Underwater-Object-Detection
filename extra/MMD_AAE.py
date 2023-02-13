import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class Encoder(nn.Module):
    def __init__(self,input_shape,hidden_layer = 2000):
        super(Encoder,self).__init__()
        self.fc = nn.Linear(input_shape,hidden_layer)
        return

    def forward(self,x):
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_shape, hidden_layer=2000):
        super(Decoder, self).__init__()
        # self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(hidden_layer,hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, input_shape)
        return

    def forward(self, x):
        # x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Taskout(nn.Module):
    def __init__(self, n_class, hidden_layer=2000):
        super(Taskout, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(hidden_layer,hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, n_class)
        return

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x,dim=1)
        return x

class Adversarial(nn.Module):
    def __init__(self, hidden_layer=2000):
        super(Adversarial, self).__init__()
        self.fc1 = nn.Linear(hidden_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class MMD_AAE(nn.Module):
    def __init__(self, input_shape, nClass):
        super(MMD_AAE,self).__init__()
        self.input_shape = input_shape
        self.E = Encoder(input_shape = input_shape, hidden_layer=2000)
        self.D = Decoder(input_shape = input_shape, hidden_layer=2000)
        self.T = Taskout(n_class = nClass, hidden_layer=2000)
        return

    def forward(self,x):
        e = self.E(x)
        d = self.D(e)
        t = self.T(e)

        return e,d,t


def MMD_Loss_func(num_source, sigmas=None):
    if sigmas is None:
        sigmas = [1, 5, 10]
    def loss(e_pred,d_ture):
        cost = 0.0
        for i in range(num_source):
            domain_i = e_pred[d_ture == i]
            if domain_i.shape[0] != 0:
                for j in range(i+1,num_source):
                    domain_j = e_pred[d_ture == j]
                    if domain_j.shape[0] != 0:
                        single_res = mmd_two_distribution(domain_i,domain_j,sigmas=sigmas)
                        cost += single_res
        return cost
    return loss

def mmd_two_distribution(source, target, sigmas):
    sigmas = torch.tensor(sigmas).cuda()
    xy = rbf_kernel(source, target, sigmas)
    xx = rbf_kernel(source, source, sigmas)
    yy = rbf_kernel(target, target, sigmas)
    return xx + yy - 2 * xy

def rbf_kernel(x, y, sigmas):
    sigmas = sigmas.reshape(sigmas.shape + (1,))
    beta = 1. / (2. * sigmas)
    dist = compute_pairwise_distances(x, y)
    dot = -torch.matmul(beta, torch.reshape(dist, (1, -1)))
    exp = torch.mean(torch.exp(dot))
    return exp

def compute_pairwise_distances(x, y):
    dist = torch.zeros(x.size(0),y.size(0)).cuda()
    for i in range(x.size(0)):
        dist[i,:] = torch.sum(torch.square(x[i].expand(y.shape) - y),dim=1)
    return dist