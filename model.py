import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
import torch.optim as optim

class Conv2dMod(nn.Module):
    """Some Information about Conv2dMod"""
    def __init__(self, input_channels, output_channels, kernel_size, eps=1e-8):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size, dtype=torch.float16))
        nn.init.xavier_uniform_(self.weight) # initialize weight
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x, y):
        # x: (batch_size, input_channels, H, W) 
        # y: (batch_size, output_channels)
        # self.weight: (output_channels, input_channels, kernel_size, kernel_size)
        N, C, H, W = x.shape
        
        # reshape weight
        w1 = y[:, None, :, None, None]
        w1 = w1.swapaxes(1, 2)
        w2 = self.weight[None, :, :, :, :]
        # modulate
        weight = w1 * w2

        # demodulate
        d = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
        weight = weight * d
        # weight: (batch_size, output_channels, input_channels, kernel_size, kernel_size)
        
        # convolution
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.shape
        weight = weight.reshape(self.output_channels * N, *ws)
        x = F.conv2d(x, weight, stride=1, padding=1, groups=N)
        x = x.reshape(N, self.output_channels, H, W)

        return x

class Bias2d(nn.Module):
    """Some Information about Bias2d"""
    def __init__(self, channel, height, width):
        super(Bias2d, self).__init__()
        self.bias = nn.Parameter(torch.randn(channel, height, width, dtype=torch.float16))
    def forward(self, x):
        return x + self.bias
    
class NoiseInjection(nn.Module):
    """Some Information about NoiseInjection"""
    def __init__(self):
        super(NoiseInjection, self).__init__()
        self.gain = nn.Parameter(torch.ones((), dtype=torch.float16))
    def forward(self, x):
        return x + torch.randn_like(x) * self.gain

