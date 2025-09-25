import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P


class ZeroMean(nn.Module):
    """Enforce zero mean kernels for each output channel"""

    def forward(self, x):
        return x - x.mean(dim=(1, 2, 3), keepdim=True)


class MultiConv2d(nn.Module):
    def __init__(self, num_channels, size_kernels):
        """"Class for multiple convolutional layers with spectral normalization"""
        super().__init__()

        self.size_kernels = size_kernels
        self.num_channels = num_channels

        # list of convolutionnal layers
        self.conv_layers = nn.ModuleList()
        for j in range(len(num_channels) - 1):
            self.conv_layers.append(nn.Conv2d(in_channels=num_channels[j], out_channels=num_channels[j+1],
                                              kernel_size=size_kernels[j], padding=size_kernels[j]//2, bias=False))
        P.register_parametrization(self.conv_layers[0], "weight", ZeroMean())

        # cache the estimation of the spectral norm
        self.L = torch.tensor(1., requires_grad=True)
        self.padding_total = sum(
            [kernel_size//2 for kernel_size in size_kernels])
        self.dirac = torch.zeros(
            (1, 1) + (4 * self.padding_total + 1, 4 * self.padding_total + 1))
        self.dirac[0, 0, 2*self.padding_total, 2 * self.padding_total] = 1

    def forward(self, x):
        x = x / torch.sqrt(self.L)
        for conv in self.conv_layers:
            x = F.conv2d(x, conv.weight, padding=conv.padding)

        return x

    def transpose(self, x):
        x = x / torch.sqrt(self.L)
        for conv in reversed(self.conv_layers):
            x = F.conv_transpose2d(x, conv.weight, padding=conv.padding)

        return x

    def spectral_norm(self, mode="fourier", n_steps=500):
        """ Compute the spectral norm of the convolutional layer
            Args:
                mode:
                    - "fourier" computes spectral norm using the DFT of the equivalent convolutional kernel. 
                    This is only an estimate (boundary effects are not taken into account) but it is differentiable and fast
                    - "power_method" computes the spectral norm by power iteration. This is more accurate and used before testing
                n_steps: number of steps for the power method
        """

        if mode == 'fourier':
            # temporary set L to 1 to get the spectral norm of the unnormalized filter
            self.L = torch.tensor(1., device=self.dirac.device)
            kernel = self.get_kernel_WtW()
            self.L = torch.fft.fft2(kernel, s=[256, 256]).abs().max()

        elif mode == 'power_method':
            self.L = torch.tensor(1., device=self.dirac.device)
            u = torch.randn((1, 1, 256, 256), device=self.dirac.device)
            with torch.no_grad():
                for _ in range(n_steps):
                    u = self.transpose(self.forward(u))
                    u = u / torch.linalg.norm(u)
            self.L = torch.linalg.norm(self.transpose(self.forward(u)))

        return self.L

    def get_kernel_WtW(self):
        impulse = self.forward(self.dirac)
        impulse = self.transpose(impulse)
        return impulse

    def _apply(self, fn):
        self.dirac = fn(self.dirac)
        return super()._apply(fn)
