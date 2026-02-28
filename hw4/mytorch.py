import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.W = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))

        if self.bias:
            self.b = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.b)
        else:
            self.b = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """
        batch_size, C, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Unfold input: (batch_size, C * k * k, H_out * W_out)
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # Reshape weight: (out_channels, C * k * k)
        W_flat = self.W.reshape(self.out_channels, -1)

        # Forward: (batch_size, out_channels, H_out * W_out)
        out = W_flat @ x_unfold

        if self.b is not None:
            out = out + self.b.reshape(1, -1, 1)

        # Reshape: (batch_size, out_channels, H_out, W_out)
        return out.reshape(batch_size, self.out_channels, H_out, W_out)


class MyMaxPool2D(nn.Module):
    def __init__(self, kernel_size, stride=None):
        """
        My custom MaxPooling 2D layer.

        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """
        batch_size, channel, input_height, input_width = x.shape

        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1

        # Reshape for unfold to work per-channel
        x_reshaped = x.reshape(batch_size * channel, 1, input_height, input_width)

        # Unfold: (batch_size * channel, k*k, output_height * output_width)
        x_unfold = F.unfold(x_reshaped, kernel_size=self.kernel_size, stride=self.stride)

        # Max over the kernel dimension
        x_pool_out, _ = x_unfold.max(dim=1)

        x_pool_out = x_pool_out.reshape(batch_size, channel, output_height, output_width)
        return x_pool_out


if __name__ == "__main__":
    from utils import set_seed
    set_seed(42)

    print("[Test 1.1] MyConv2D: basic forward (3->8, k=3, s=1, p=1)")
    my_conv1 = MyConv2D(3, 8, kernel_size=3, stride=1, padding=1, bias=True)
    pt_conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)

    pt_conv1.weight = nn.Parameter(my_conv1.W.data.clone())
    pt_conv1.bias = nn.Parameter(my_conv1.b.data.clone())

    x1 = torch.randn(2, 3, 16, 16)
    out_my1 = my_conv1(x1)
    out_pt1 = pt_conv1(x1)
    err1 = (out_my1 - out_pt1).abs().max().item()
    print(f"  {'PASS' if err1 < 1e-5 else 'FAIL'} with max error: {err1:.2e}")

    print("\n[Test 1.2] MyConv2D: stride=2, padding=2, bias=False")
    my_conv2 = MyConv2D(1, 4, kernel_size=5, stride=2, padding=2, bias=False)
    pt_conv2 = nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=2, bias=False)

    pt_conv2.weight = nn.Parameter(my_conv2.W.data.clone())

    x2 = torch.randn(4, 1, 32, 32)
    out_my2 = my_conv2(x2)
    out_pt2 = pt_conv2(x2)
    err2 = (out_my2 - out_pt2).abs().max().item()
    print(f"  Output shape: {out_my2.shape} (expected: {out_pt2.shape})")
    print(f"  {'PASS' if err2 < 1e-5 else 'FAIL'} with max error: {err2:.2e}")

    print("\n[Test 2.1] MyMaxPool2D: kernel_size=2, stride=2")
    my_pool3 = MyMaxPool2D(kernel_size=2, stride=2)
    pt_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    x3 = torch.randn(2, 3, 8, 8)
    out_my3 = my_pool3(x3)
    out_pt3 = pt_pool3(x3)
    err3 = (out_my3 - out_pt3).abs().max().item()
    print(f"  Output shape: {out_my3.shape} (expected: {out_pt3.shape})")
    print(f"  {'PASS' if err3 < 1e-5 else 'FAIL'} with max error: {err3:.2e}")

    print("\n[Test 2.2] MyMaxPool2D: kernel_size=3, stride=1")
    my_pool2 = MyMaxPool2D(kernel_size=3, stride=1)
    pt_pool2 = nn.MaxPool2d(kernel_size=3, stride=1)

    x4 = torch.randn(1, 16, 10, 10)
    out_my4 = my_pool2(x4)
    out_pt4 = pt_pool2(x4)
    err4 = (out_my4 - out_pt4).abs().max().item()
    print(f"  Output shape: {out_my4.shape} (expected: {out_pt4.shape})")
    print(f"  {'PASS' if err4 < 1e-5 else 'FAIL'} with max error: {err4:.2e}")