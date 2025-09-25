import torch
import torch.nn as nn
import torch.nn.functional as F


class TVFilters(nn.Module):
    """Class for total variation filters"""
    def __init__(self):
        super().__init__()

        self.filter1 = torch.Tensor([[[[1., -1]]]]) / torch.sqrt(torch.tensor(8.))
        self.filter2 = torch.Tensor([[[[1], [-1]]]]) / torch.sqrt(torch.tensor(8.))

    def forward(self, x):

        L1 = F.pad(F.conv2d(x, self.filter1), (0, 1), "constant", 0)
        L2 = F.pad(F.conv2d(x, self.filter2), (0, 0, 0, 1), "constant", 0)
        Lx = torch.cat((L1, L2), dim=1)
        return Lx

    def transpose(self, x):

        L1t = F.conv_transpose2d(x[:, 0:1, :, :-1], self.filter1)
        L2t = F.conv_transpose2d(x[:, 1:2, :-1, :], self.filter2)
        Ltx = L1t + L2t
        return Ltx
    
    def _apply(self, fn):
        self.filter1 = fn(self.filter1)
        self.filter2 = fn(self.filter2)
        return super()._apply(fn)
    