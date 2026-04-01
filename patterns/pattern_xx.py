import torch

from .pattern_base import pattern_base


class pattern_xx(pattern_base):
    def __init__(
        self,
        size_x,
        size_y,
        Nx,
        Ny,
        lamb0,
        CRA,
        azimuth,
        eps_bg,
        eps_obj,
        radius=None,
        width=None,
        height=None,
        center_x=None,
        center_y=None,
        theta=0.0,
        sharpness=1000.0,
        dtype_f=torch.float64,
        dtype_c=torch.complex128,
        device='cpu',
    ):
        super().__init__(
            size_x=size_x,
            size_y=size_y,
            Nx=Nx,
            Ny=Ny,
            lamb0=lamb0,
            CRA=CRA,
            azimuth=azimuth,
            sharpness=sharpness,
            dtype_f=dtype_f,
            dtype_c=dtype_c,
            device=device,
        )
        self.eps_bg = torch.as_tensor(eps_bg, dtype=self.dtype_c, device=self.device)
        self.eps_obj = torch.as_tensor(eps_obj, dtype=self.dtype_c, device=self.device)
        self.radius = radius
        self.width = width
        self.height = height
        self.center_x = size_x / 2 if center_x is None else center_x
        self.center_y = size_y / 2 if center_y is None else center_y
        self.theta = theta

    def build_mask(self):
        if self.radius is not None:
            return self.circle(self.radius, self.center_x, self.center_y)

        if self.width is not None and self.height is not None:
            return self.rectangle(self.width, self.height, self.center_x, self.center_y, self.theta)

        raise ValueError('Set either radius or both width and height')

    def build_eps(self):
        mask = self.build_mask()
        return self.blend_eps(self.eps_bg, self.eps_obj, mask)

    def flatten(self):
        return self.flatten_eps(self.build_eps())
