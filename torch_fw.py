import torch
from torch import zeros, cat, bmm
import torch.nn as nn
from torch.nn import Linear, GRU, Embedding, Module, NLLLoss, Dropout
from torch import optim, LongTensor
from torch.optim import SGD
from torch.nn.functional import softmax, log_softmax, relu


class Variable(torch.autograd.Variable):
    pass


def manual_seed(seed):
    torch.manual_seed(seed)