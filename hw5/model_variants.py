# CMU 18-780/6 Homework 4
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
# For the assignment, you are asked to create the architectures of these
# three networks by filling in the __init__ and forward methods in the
# DCGenerator, DCDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn
import torch.nn.functional as F

def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Create a convolutional layer, with optional normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

def l2_normalize(x, eps=1e-12):
    """Numerically stable vector normalization used by spectral norm."""
    return x / (x.norm() + eps)


class SpectralNormConv2d(nn.Module):
    """Conv2d layer with manual spectral normalization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, n_power_iterations=1, eps=1e-12):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.register_buffer('u', l2_normalize(torch.randn(out_channels), eps))

    def _get_spectral_normalized_weight(self):
        weight = self.conv.weight
        weight_mat = weight.view(weight.size(0), -1)
        u = self.u

        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = l2_normalize(torch.mv(weight_mat.t(), u), self.eps)
                u = l2_normalize(torch.mv(weight_mat, v), self.eps)
            if self.training:
                self.u.copy_(u)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        return weight / sigma.clamp_min(self.eps)

    def forward(self, x):
        weight = self._get_spectral_normalized_weight()
        return F.conv2d(x, weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)


def sn_conv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm=None, activ=None):
    """Create a spectrally normalized convolutional layer."""
    layers = []
    layers.append(SpectralNormConv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    ))

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim=64):
        super().__init__()

        self.up_conv1 = conv(noise_size, conv_dim * 8, 4, 1, 3, norm='instance', activ='relu')
        self.up_conv2 = up_conv(conv_dim * 8, conv_dim * 4, 3, 1, 1, 2, norm='instance', activ='relu')
        self.up_conv3 = up_conv(conv_dim * 4, conv_dim * 2, 3, 1, 1, 2, norm='instance', activ='relu')
        self.up_conv4 = up_conv(conv_dim * 2, conv_dim, 3, 1, 1, 2, norm='instance', activ='relu')
        self.up_conv5 = up_conv(conv_dim, 3, 3, 1, 1, 2, norm=None, activ='tanh')

    def forward(self, z):
        """
        Generate an image given a sample of random noise.

        Input
        -----
            z: BS x noise_size x 1 x 1   -->  16x100x1x1

        Output
        ------
            out: BS x channels x image_width x image_height  -->  16x3x64x64
        """
        z = self.up_conv1(z)
        z = self.up_conv2(z)
        z = self.up_conv3(z)
        z = self.up_conv4(z)
        z = self.up_conv5(z)
        return z


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1, norm=norm,
            activ=activ
        )

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class DCDiscriminator(nn.Module):
    """Architecture of the discriminator network."""
    def __init__(self, conv_dim=64, norm='instance'):
        super().__init__()
        self.conv1 = conv(3, conv_dim, 4, 2, 1, norm, False, 'relu')
        self.conv2 = conv(conv_dim, conv_dim * 2, 4, 2, 1, norm, False, 'relu')
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4, 2, 1, norm, False, 'relu')
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4, 2, 1, norm, False, 'relu')
        self.conv5 = conv(conv_dim * 8, 1, 4, 1, 0, norm=None, init_zero_weights=True)

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()


class SNDCDiscriminator(nn.Module):
    """DCGAN discriminator with spectral normalization on convolution layers."""
    def __init__(self, conv_dim=64, norm=None):
        super().__init__()
        self.conv1 = sn_conv(3, conv_dim, 4, 2, 1, norm, 'relu')
        self.conv2 = sn_conv(conv_dim, conv_dim * 2, 4, 2, 1, norm, 'relu')
        self.conv3 = sn_conv(conv_dim * 2, conv_dim * 4, 4, 2, 1, norm, 'relu')
        self.conv4 = sn_conv(conv_dim * 4, conv_dim * 8, 4, 2, 1, norm, 'relu')
        self.conv5 = sn_conv(conv_dim * 8, 1, 4, 1, 0, norm=None)

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()
