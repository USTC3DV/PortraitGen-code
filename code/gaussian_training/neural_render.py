import torch
import torch.nn as nn
import torch.nn.functional as F


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        y = self.act_fn(self.conv(x))
        return y

class ConvReLU(nn.Module):
    """A module that encapsulates Convolution, Instance Normalization, and ReLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.norm = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class Neural_Renderer(nn.Module):
    def __init__(self, input_dim=32, output_dim=3, network_capacity=32):
        super(Neural_Renderer, self).__init__()
        self.convs = nn.Sequential(
            ConvReLU(input_dim, network_capacity*4),
            ConvReLU(network_capacity*4, network_capacity*2),
            ConvReLU(network_capacity*2, network_capacity),
            ConvReLU(network_capacity, network_capacity),
            ConvReLU(network_capacity, network_capacity),
            OutConv(network_capacity, output_dim)
        )


    def forward(self, x):
        x =self.convs(x)
        return x