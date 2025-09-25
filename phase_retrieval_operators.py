import torch
import torch.nn as nn
torch.manual_seed(0)

class RandomPhaseRetrieval(nn.Module):

    def __init__(self, p, img_shape, nb_layers=1):
        super(RandomPhaseRetrieval, self).__init__()

        self.img_shape = img_shape
        self.nb_layers = nb_layers

        self.diags = [(2 * torch.randint(0, 2, img_shape) - 1) for _ in range(nb_layers)]
        self.mask = torch.bernoulli(p * torch.ones(img_shape))

    def forward(self, x):
        return self.measurement(x)

    def A(self, x):
        if self.nb_layers == 0:
            return self.mask * torch.fft.fft2(x, norm='ortho')
        for diag in self.diags:
            x = torch.fft.fft2(diag * x, norm='ortho')
        x = self.mask * x
        return x
    
    def A_adjoint(self, x):
        if self.nb_layers == 0:
            return torch.fft.ifft2(self.mask * x, norm='ortho')
        x = x * self.mask
        for diag in reversed(self.diags):
            x = diag.conj() * torch.fft.ifft2(x, norm='ortho')
        return x
        
    def measurement(self, x):
        return self.A(x).abs()
    
    def cost(self, x, y):
            return torch.norm(self.measurement(x) - y) ** 2
        
    def grad(self, x, y):
        return self.A_adjoint(torch.exp(1j * self.A(x).angle()) * (self.measurement(x) - y))
    
    def prox(self, x, y, mu):
        # compute Ux
        Ux = x.clone()
        if self.nb_layers == 0:
            Ux = torch.fft.fft2(Ux, norm='ortho')
        else:
            for diag in self.diags:
                Ux = torch.fft.fft2(diag * Ux, norm='ortho')

        # compute new amplitude
        Ux_abs = torch.abs(Ux)
        Ux_angle = torch.angle(Ux)
        new_abs = self.mask * (y + mu * Ux_abs) / (1 + mu) + (1 - self.mask) * Ux_abs
        z = new_abs * torch.exp(1j * Ux_angle)

        # compute U^H z
        if self.nb_layers == 0:
            z = torch.fft.ifft2(z, norm='ortho')
        else:
            for diag in reversed(self.diags):
                z = diag.conj() * torch.fft.ifft2(z, norm='ortho')
        return z

    def _apply(self, fn):
        self.mask = fn(self.mask)
        for k in range(len(self.diags)):
            self.diags[k] = fn(self.diags[k])
        return super()._apply(fn)
    