import torch
from torch import zeros, cat, bmm, manual_seed
from torch.nn import Linear, GRU, Embedding, Module, NLLLoss, Dropout
from torch import optim, LongTensor
from torch.optim import SGD
from torch.autograd import Variable
from torch.nn.functional import softmax, log_softmax, relu
from torch.nn import LogSoftmax