# Use with pytorch version >= 1.1.0

import torch
import torch.nn as nn
from torch.nn import functional as F


class GaborConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.register_parameter('weight', nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1], requires_grad=False)))
        self.register_parameter('freq', nn.Parameter(
            (3.14 / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor)))
        self.register_parameter('theta', nn.Parameter(
            (3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor)))
        self.register_parameter('psi', nn.Parameter(3.14 * torch.rand(out_channels, in_channels)))
        self.register_parameter('sigma', nn.Parameter(3.14 / self.freq))

        x0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        y0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]
        y, x = torch.meshgrid([torch.linspace(-x0 + 1, x0, kernel_size[0]),
                               torch.linspace(-y0 + 1, y0, kernel_size[1])])
        self.y = nn.Parameter(y, requires_grad=False)
        self.x = nn.Parameter(x, requires_grad=False)

    def forward(self, input_image):
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + 1e-3) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * 3.14 * sigma ** 2)
                self.weight[i, j] = g
        return F.conv2d(input_image, self.weight)
