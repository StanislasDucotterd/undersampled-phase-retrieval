import torch

def APGDR(x_init, y, model, sigma, physics, n_sigma, max_iter=10000, tol=1e-5):

    # initial value: noisy image
    x = torch.clone(x_init)
    z = x.clone()
    t = torch.ones(x.shape[0], device=x.device).view(-1, 1, 1, 1)

    # cache values of scaling coeff for efficiency
    scaling = model.get_scaling(sigma=sigma)

    # the index of the images that have not converged yet
    idx = torch.arange(0, x.shape[0], device=x.device)
    # relative change in the estimate
    res = torch.ones(x.shape[0], device=x.device)
    old_cost = 1e12*torch.ones(x.shape[0], device=x.device)
        
    for i in range(max_iter):
        model.scaling = scaling[idx]
        x_old = torch.clone(x)
        grad_abs, cost_abs = model.grad_cost(torch.abs(z[idx]), sigma[idx])
        grad_phase, cost_phase = model.grad_cost(torch.angle(z[idx]) / (2 * torch.pi), sigma[idx])
        grad = torch.exp(1j * torch.angle(z[idx])) * grad_abs \
                + 1j * z * grad_phase / (2 * torch.pi * z[idx].abs()**2 + 1e-8)
        grad = grad / model.lip()
        cost = cost_abs + cost_phase
        if n_sigma > 0:
            cost = cost + physics.cost(z, y) / (2 * n_sigma)
        x[idx] = physics.prox(z[idx] - grad, y[idx], model.lip() * n_sigma)

        t_old = torch.clone(t)
        t = 0.5 * (1 + torch.sqrt(1 + 4*t**2))
        z[idx] = x[idx] + (t_old[idx] - 1)/t[idx] * (x[idx] - x_old[idx])

        if i > 0:
            res[idx] = torch.norm(
                x[idx] - x_old[idx], p=2, dim=(1, 2, 3)) / (torch.norm(x[idx], p=2, dim=(1, 2, 3)))
        print(res.item(), cost.item())

        esti = cost - old_cost[idx]
        old_cost[idx] = cost
        id_restart = (esti > 0).nonzero().view(-1)
        t[idx[id_restart]] = 1
        z[idx[id_restart]] = x[idx[id_restart]]

        condition = (res > tol)
        idx = condition.nonzero().view(-1)

        if condition.sum() == 0:
            break

    model.clear_cache()

    return x


def proj_l1_channel(x):
    """
    Projects a batch of vectors x in dimension d onto the unit l1-ball.
    The dimensions d and K are defined in the paper, H and W are the height and width of the image.

    Args:
        x torch.Tensor: Input tensor of shape (batch, d, K, H, W).
    """
    norm_x = torch.norm(x, p=1, dim=1, keepdim=True)
    mask = (norm_x <= 1.0).repeat(1, x.shape[1], 1, 1, 1)

    if mask.all():
        return x  # Already within the l1-ball

    abs_x = torch.abs(x)
    sorted_x, _ = torch.sort(abs_x, descending=True, dim=1)
    cumsum_x = torch.cumsum(sorted_x, dim=1)

    input_dim = x.shape[1]
    rho = (sorted_x * torch.arange(1, input_dim + 1, device=x.device)
           [None, :, None, None, None] > (cumsum_x - 1.0)).sum(dim=1) - 1

    # This is necessary due to numerical errors
    rho[rho < 0] = 0

    theta = (torch.gather(cumsum_x, 1, rho.unsqueeze(1)) - 1.0) / \
        (rho + 1).unsqueeze(1)
    projected_x = torch.sign(x) * torch.clamp(abs_x - theta, min=0)
    projected_x[mask] = x[mask]

    return projected_x

def correct_global_phase(pred, ground_truth):
    phase = torch.sum(torch.conj(pred) * ground_truth) / (torch.sum(torch.abs(ground_truth)**2) + 1e-8)
    return pred * phase

def cosine_similarity(pred, ground_truth):
    scalar = (pred * ground_truth.conj()).sum()
    return scalar.abs() / (pred.norm() * ground_truth.norm())

def transform(x, iter):
    if iter % 8 == 1:
        return torch.flip(x, [2])
    elif iter % 8 == 2:
        return torch.flip(x, [3])
    elif iter % 8 == 3:
        return torch.rot90(x, 1, [2, 3])
    elif iter % 8 == 4:
        return torch.rot90(x, 2, [2, 3])
    elif iter % 8 == 5:
        return torch.rot90(x, 3, [2, 3])
    elif iter % 8 == 6:
        return torch.flip(torch.rot90(x, 1, [2, 3]), [2])
    elif iter % 8 == 7:
        return torch.flip(torch.rot90(x, 1, [2, 3]), [3])
    else:
        return x
    
def inverse(x, iter):
    if iter % 8 == 1:
        return torch.flip(x, [2])
    elif iter % 8 == 2:
        return torch.flip(x, [3])
    elif iter % 8 == 3:
        return torch.rot90(x, 3, [2, 3])
    elif iter % 8 == 4:
        return torch.rot90(x, 2, [2, 3])
    elif iter % 8 == 5:
        return torch.rot90(x, 1, [2, 3])
    elif iter % 8 == 6:
        return torch.rot90(torch.flip(x, [2]), 3, [2, 3])
    elif iter % 8 == 7:
        return torch.rot90(torch.flip(x, [3]), 3, [2, 3])
    else:
        return x
