import torch


def t_grad(fun,x,dx,ind):
    if not torch.is_tensor(x):
        raise TypeError('t_grad expects a torch.Tensor input')

    dx = float(dx)
    base = x.detach().clone()

    if base.ndim == 0:
        y1 = fun(base)
        y2 = fun(base + dx)

        x_ad = (base + 0.5*dx).clone().requires_grad_(True)
        y = fun(x_ad)
        (g,) = torch.autograd.grad(y, x_ad)
        return (y2-y1)/dx, g

    y1 = fun(base)
    x_shift = base.clone()
    x_shift.reshape(-1)[ind] += dx
    y2 = fun(x_shift)

    x_ad = base.clone()
    x_ad.reshape(-1)[ind] += 0.5*dx
    x_ad = x_ad.requires_grad_(True)
    y = fun(x_ad)
    (g,) = torch.autograd.grad(y, x_ad)

    return (y2-y1)/dx, g.reshape(-1)[ind]

