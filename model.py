import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
import torch.optim as optim

class Conv2dMod(nn.Module):
    """Some Information about Conv2dMod"""
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-8):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size))
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
        
        # reshape
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.shape
        weight = weight.reshape(self.output_channels * N, *ws)
        
        
        # padding
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), mode='replicate')
        
        # convolution
        x = F.conv2d(x, weight, stride=1, padding=0, groups=N)
        x = x.reshape(N, self.output_channels, H, W)

        return x
    
class Bias(nn.Module):
    """Some Information about Noise"""
    def __init__(self, channels):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.randn(channels))

    def forward(self, x):
        # x: (batch_size, channels, H, W)
        # self.noise: (channels)
        
        # add bias
        bias = self.bias[None, :, None, None]
        bias = bias.expand(x.shape[0], *bias.shape)
        x = x + bias
        return x

class ToRGB(nn.Module):
    """Some Information about ToRGB"""
    def __init__(self,channels):
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.conv(x)

class GeneratorBlock(nn.Module):
    """Some Information about GeneratorBlock"""
    def __init__(self, input_channels, latent_channels, output_channels, style_dim):
        super(GeneratorBlock, self).__init__()
        self.affine1 = nn.Linear(style_dim, latent_channels)
        self.conv1 = Conv2dMod(input_channels, latent_channels)
        self.bias1 = Bias(latent_channels)
        self.activation1 = nn.LeakyReLU()
        
        self.affine2 = nn.Linear(style_dim, output_channels)
        self.conv2 = Conv2dMod(latent_channels, output_channels)
        self.bias2 = Bias(output_channels)
        self.activation1 = nn.LeakyReLU()
        
        self.to_rgb = ToRGB(output_channels)
    def forward(self, x):
        x = self.conv1(x, self.affine1(x))
        x = self.bias1(x)
        x = self.activation1(x)
        
        x = self.conv2(x, self.affine2(x))
        x = self.bias2(x)
        x = self.activation1(x)
        rgb = self.to_rgb(x)
        
        return x, rgb

