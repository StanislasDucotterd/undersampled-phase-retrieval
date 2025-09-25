import torch
import torch.nn as nn
from models.optimization import proj_l1_channel
from models.multi_conv import MultiConv2d
import torch.nn.functional as F


class MFoE(nn.Module):
    """
    Multivariate Fields of Experts (MFoE) model.

    This model implements a multivariate extension of the Fields of Experts framework
    with potential functions based on the Moreau envelope of the ℓ∞ norm.
    """

    def __init__(self, param_model, param_multi_conv, param_fw, param_bw):
        super().__init__()

        # 1 - multi-convolutionnal layers
        print("Multi convolutionnal layer: ", param_multi_conv)
        self.conv_layer = MultiConv2d(**param_multi_conv)
        self.nb_filters = param_multi_conv['num_channels'][-1]
        # Corresponds to d in the paper
        self.groupsize = param_model['groupsize']
        # Corresponds to K in the paper
        self.nb_groups = self.nb_filters // self.groupsize

        # 2 - Activation parameters
        self.convex = param_model['convex']
        self.lamb = nn.Parameter(torch.tensor(param_model['lamb_init']))
        self.Q_param = nn.Parameter(torch.rand(
            self.nb_groups, self.groupsize, self.groupsize) - 0.5)
        self.taus = nn.Parameter(
            param_model['lamb_init']/2*torch.ones(1, 1, self.nb_groups, 1, 1))
        self.scale = 0.999

        # 3 - mu function to add some flexibility to the activation accross channels and noise levels
        self.slope = param_model['scaling']
        self.mu = nn.Sequential(*[nn.Linear(1, self.nb_groups), nn.ReLU(), nn.Linear(self.nb_groups, self.nb_groups),
                                  nn.ReLU(), nn.Linear(self.nb_groups, self.nb_groups)])

        # 4 - forward and backward hyperparameters for the DEQ
        self.param_fw = param_fw
        self.param_bw = param_bw

        self.num_params = sum(p.numel() for p in self.parameters())

        # Parameters to cache
        self.scaling = None
        self.Q = None
        self.Q_norms = None

    def get_scaling(self, sigma):
        if self.scaling is None:
            sigma = sigma[:, :, 0, 0]
            scaling = F.relu(self.mu(sigma*20 - 2)*0.05 +
                             sigma) * self.slope + 1e-9
            scaling = scaling.view(-1, 1, self.nb_groups, 1, 1)
            return scaling
        else:
            return self.scaling

    def clear_cache(self):
        self.scaling = None
        self.Q = None
        self.Q_norms = None

    def orient(self, x):
        "Apply the matrix Q"
        if self.groupsize > 1:
            if self.Q is None:
                self.Q = self.Q_param / \
                    torch.max(torch.sum(self.Q_param.abs(), dim=2,
                              keepdim=True), torch.tensor(1.0))
                self.Q_norms = torch.linalg.matrix_norm(
                    self.Q, ord=2, dim=(1, 2), keepdim=True)
            Q = self.Q / self.Q_norms**2
            return self.scale * torch.einsum('flg,bgfhw->blfhw', Q, x)
        else:
            return self.scale * x

    def unorient(self, x):
        "Apply the matrix Q^T"
        if self.groupsize > 1:
            return self.scale * torch.einsum('flg,bgfhw->blfhw', self.Q.transpose(1, 2), x)
        else:
            return self.scale * x

    def moreau(self, x):
        if self.groupsize > 1:
            grad = proj_l1_channel(x)
        else:
            grad = torch.clip(x, -1., 1.)
        cost = (x - grad).norm(dim=1, p=float('inf')) + \
            (1/2)*grad.norm(dim=1, p=2)**2
        cost = cost.unsqueeze(1)
        return grad, cost

    def activation(self, x):
        grad_convex, cost_convex = self.moreau(x)
        if self.convex:
            grad_concave, cost_concave = 0., 0.
        else:
            taus = F.relu(self.taus).exp()
            grad_concave, cost_concave = self.moreau(self.orient(x) / taus)
            grad_concave = self.unorient(grad_concave)
            cost_concave = taus * cost_concave
            if self.groupsize > 1:
                cost_concave = self.Q_norms**2 * cost_concave
        grad = self.lamb.exp() * (grad_convex - grad_concave)
        cost = self.lamb.exp() * (cost_convex - cost_concave)
        return grad, cost

    def grad_cost(self, x, sigma):
        # Applying W
        Wx = self.conv_layer(x)

        # Applying nonlinearity
        Wx = Wx.view(-1, self.groupsize, self.nb_groups, *x.shape[2:])
        scaling = self.get_scaling(sigma)
        Wx = Wx / scaling
        grad, cost = self.activation(Wx)
        grad = grad * scaling
        grad = grad.view(-1, self.nb_filters, *x.shape[2:])
        grad = self.conv_layer.transpose(grad)
        cost = cost * scaling**2
        cost = cost.sum(dim=(1, 2, 3, 4))
        return grad, cost

    def lip(self):
        return self.lamb.exp()
