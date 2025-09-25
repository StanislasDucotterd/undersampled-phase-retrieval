import torch
import torch.nn as nn
from models.tv_filters import TVFilters


class MoreauTV(nn.Module):
    def __init__(self, lamb=0.0, mu=0.0):
        super().__init__()
        r"""
        Moreau envelope of isotropic TV.

        Parameters
        ----------
        lamb : float, optional (default=0.0)
            Log-scaled regularization weight 
        mu : float, optional (default=0.0)
            Log-scaled parameter of the Moreau envelope.
        """
        self.tv_filters = TVFilters()
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.mu = nn.Parameter(torch.tensor(mu))

    def get_scaling(self, sigma=None):
        return sigma
    
    def clear_cache(self):
        pass

    def moreau(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        grad = torch.clip(x_norm, -1., 1.) * x / (x_norm + 1e-8)    
        cost = (x - grad).norm(dim=1, p=self.norm) + (1/2)*grad.norm(dim=1, p=2)**2
        return grad, cost

    def grad_cost(self, x, sigma):
        Lx = self.tv_filters(x)
        grad, cost = self.moreau(Lx / self.mu.exp())
        grad = self.lamb.exp() * self.tv_filters.transpose(grad)
        cost = self.lamb.exp() * self.mu.exp() * cost.sum()
        return grad, cost

    def lip(self):
        return self.lamb.exp() / self.mu.exp() 
