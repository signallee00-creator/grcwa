import torch

nm = 1e-9


class pattern_base:
    def __init__(
        self,
        size_x,
        size_y,
        Nx,
        Ny,
        lamb0,
        CRA,
        azimuth,
        sharpness=1000.0,
        dtype_f=torch.float64,
        dtype_c=torch.complex128,
        device='cpu',
    ):
        self.device = torch.device(device)
        self.dtype_f = dtype_f
        self.dtype_c = dtype_c

        self.size_x = torch.as_tensor(size_x, dtype=self.dtype_f, device=self.device)
        self.size_y = torch.as_tensor(size_y, dtype=self.dtype_f, device=self.device)
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.lamb0 = torch.as_tensor(lamb0, dtype=self.dtype_f, device=self.device)
        self.CRA = torch.as_tensor(CRA, dtype=self.dtype_f, device=self.device)
        self.azimuth = torch.as_tensor(azimuth, dtype=self.dtype_f, device=self.device) * torch.pi / 180.0

        self.grid_size_x = self.size_x / self.Nx
        self.grid_size_y = self.size_y / self.Ny
        self.grid_size = torch.minimum(self.grid_size_x, self.grid_size_y)

        sharpness = torch.as_tensor(sharpness, dtype=self.dtype_f, device=self.device)
        self.edge_width = self.grid_size / sharpness * 1000.0
        self.sigmoid_on = bool(sharpness.item() > 0.0 and self.edge_width.item() > 0.0)

        self.x = None
        self.y = None
        self.x_grid = None
        self.y_grid = None

    def grid(self):
        dx = self.size_x / self.Nx
        dy = self.size_y / self.Ny

        self.x = (torch.arange(self.Nx, dtype=self.dtype_f, device=self.device) - (self.Nx - 1) / 2) * dx + self.size_x / 2
        self.y = (torch.arange(self.Ny, dtype=self.dtype_f, device=self.device) - (self.Ny - 1) / 2) * dy + self.size_y / 2
        self.x_grid, self.y_grid = torch.meshgrid(self.x, self.y, indexing='ij')

    def ensure_grid(self):
        if self.x_grid is None or self.y_grid is None:
            self.grid()

    def _mask_from_sd(self, sd):
        if self.sigmoid_on:
            edge_width = torch.clamp(self.edge_width, min=torch.as_tensor(1e-12, dtype=self.dtype_f, device=self.device))
            return torch.sigmoid(sd / edge_width).to(self.dtype_f)
        return (sd >= 0).to(self.dtype_f)

    def _rotate(self, Cx, Cy, theta):
        self.ensure_grid()
        Cx = torch.as_tensor(Cx, dtype=self.dtype_f, device=self.device)
        Cy = torch.as_tensor(Cy, dtype=self.dtype_f, device=self.device)
        theta = torch.as_tensor(theta, dtype=self.dtype_f, device=self.device)

        x_local = self.x_grid - Cx
        y_local = self.y_grid - Cy

        c = torch.cos(theta)
        s = torch.sin(theta)
        xr = x_local * c + y_local * s
        yr = -x_local * s + y_local * c
        return xr, yr

    def circle(self, R, Cx, Cy):
        self.ensure_grid()
        R = torch.as_tensor(R, dtype=self.dtype_f, device=self.device)
        Cx = torch.as_tensor(Cx, dtype=self.dtype_f, device=self.device)
        Cy = torch.as_tensor(Cy, dtype=self.dtype_f, device=self.device)

        sd = R - torch.sqrt((self.x_grid - Cx) ** 2 + (self.y_grid - Cy) ** 2)
        return self._mask_from_sd(sd)

    def ellipse(self, Rx, Ry, Cx, Cy, theta=0.0):
        self.ensure_grid()
        Rx = torch.as_tensor(Rx, dtype=self.dtype_f, device=self.device)
        Ry = torch.as_tensor(Ry, dtype=self.dtype_f, device=self.device)
        xr, yr = self._rotate(Cx, Cy, theta)

        q = torch.sqrt((xr / Rx) ** 2 + (yr / Ry) ** 2)
        sd = (1.0 - q) * torch.minimum(Rx, Ry)
        return self._mask_from_sd(sd)

    def square(self, W, Cx, Cy, theta=0.0):
        return self.rectangle(W, W, Cx, Cy, theta)

    def rectangle(self, Wx, Wy, Cx, Cy, theta=0.0):
        self.ensure_grid()
        Wx = torch.as_tensor(Wx, dtype=self.dtype_f, device=self.device)
        Wy = torch.as_tensor(Wy, dtype=self.dtype_f, device=self.device)
        xr, yr = self._rotate(Cx, Cy, theta)

        qx = torch.abs(xr) - Wx / 2
        qy = torch.abs(yr) - Wy / 2
        outside = torch.sqrt(torch.clamp(qx, min=0.0) ** 2 + torch.clamp(qy, min=0.0) ** 2)
        inside = torch.clamp(torch.maximum(qx, qy), max=0.0)
        sd = -(outside + inside)
        return self._mask_from_sd(sd)

    def rhombus(self, Wx, Wy, Cx, Cy, theta=0.0):
        self.ensure_grid()
        Wx = torch.as_tensor(Wx, dtype=self.dtype_f, device=self.device)
        Wy = torch.as_tensor(Wy, dtype=self.dtype_f, device=self.device)
        xr, yr = self._rotate(Cx, Cy, theta)

        q = torch.abs(xr) / (Wx / 2) + torch.abs(yr) / (Wy / 2)
        sd = (1.0 - q) * (torch.minimum(Wx, Wy) / 2)
        return self._mask_from_sd(sd)

    def super_ellipse(self, Wx, Wy, Cx, Cy, theta=0.0, power=2.0):
        self.ensure_grid()
        Wx = torch.as_tensor(Wx, dtype=self.dtype_f, device=self.device)
        Wy = torch.as_tensor(Wy, dtype=self.dtype_f, device=self.device)
        power = torch.as_tensor(power, dtype=self.dtype_f, device=self.device)
        xr, yr = self._rotate(Cx, Cy, theta)

        q = (torch.abs(xr / (Wx / 2)) ** power + torch.abs(yr / (Wy / 2)) ** power) ** (1.0 / power)
        sd = (1.0 - q) * (torch.minimum(Wx, Wy) / 2)
        return self._mask_from_sd(sd)

    def polygon(self, verts_xy, t_inset=0.0):
        self.ensure_grid()

        x = self.x_grid
        y = self.y_grid
        verts = torch.as_tensor(verts_xy, dtype=self.dtype_f, device=self.device)
        verts_next = torch.roll(verts, shifts=-1, dims=0)

        x0 = verts[:, 0][:, None, None]
        y0 = verts[:, 1][:, None, None]
        x1 = verts_next[:, 0][:, None, None]
        y1 = verts_next[:, 1][:, None, None]

        y_eps = y + (1e-15 if self.dtype_f == torch.float64 else 1e-12)
        cond_y = (y0 > y_eps) != (y1 > y_eps)
        denom = y1 - y0
        denom = torch.where(torch.abs(denom) < 1e-20, torch.ones_like(denom), denom)
        x_int = x0 + (y_eps - y0) * (x1 - x0) / denom
        inside = ((cond_y & (x < x_int)).to(torch.int32).sum(dim=0) % 2).to(self.dtype_f)

        sx = x1 - x0
        sy = y1 - y0
        seg2 = torch.clamp(sx * sx + sy * sy, min=1e-30)
        t = ((x - x0) * sx + (y - y0) * sy) / seg2
        t = torch.clamp(t, 0.0, 1.0)

        cx = x0 + t * sx
        cy = y0 + t * sy
        dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        dmin = dist.min(dim=0).values

        t_inset = torch.as_tensor(t_inset, dtype=self.dtype_f, device=self.device)
        sd = (inside * 2.0 - 1.0) * dmin - t_inset
        return self._mask_from_sd(sd)

    def gaussian(self, r, Cx, Cy):
        self.ensure_grid()
        r = torch.as_tensor(r, dtype=self.dtype_f, device=self.device)
        Cx = torch.as_tensor(Cx, dtype=self.dtype_f, device=self.device)
        Cy = torch.as_tensor(Cy, dtype=self.dtype_f, device=self.device)
        return torch.exp(-((self.x_grid - Cx) ** 2 + (self.y_grid - Cy) ** 2) / (2 * r ** 2))

    def union(self, A, B):
        return A + B - A * B

    def multiple_union(self, *tensors):
        result = tensors[0]
        for tensor in tensors[1:]:
            result = result + tensor - result * tensor
        return result

    def intersection(self, A, B):
        return A * B

    def difference(self, A, B):
        return A * (1.0 - B)

    def blend_eps(self, eps_bg, eps_obj, mask):
        eps_bg = torch.as_tensor(eps_bg, dtype=self.dtype_c, device=self.device)
        eps_obj = torch.as_tensor(eps_obj, dtype=self.dtype_c, device=self.device)
        mask = torch.as_tensor(mask, dtype=self.dtype_f, device=self.device)
        return eps_bg * (1.0 - mask) + eps_obj * mask

    def apply_material(self, eps_base, eps_obj, mask):
        return self.blend_eps(eps_base, eps_obj, mask)

    def flatten_eps(self, eps):
        if isinstance(eps, (list, tuple)):
            return [torch.as_tensor(component, device=self.device).reshape(-1) for component in eps]
        return torch.as_tensor(eps, device=self.device).reshape(-1)

