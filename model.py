import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

class Conv2dMod(nn.Module):
    """Some Information about Conv2dMod"""
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-8):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size, dtype=torch.float32))
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
        self.bias = nn.Parameter(torch.randn(channels, dtype=torch.float32))

    def forward(self, x):
        # x: (batch_size, channels, H, W)
        # self.noise: (channels)
        
        # add bias
        bias = self.bias[None, :, None, None]
        x = x + bias
        return x

class NoiseInjection(nn.Module):
    """Some Information about Noise"""
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.from_channels = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        noise_map = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], dtype=torch.float32).to(x.device)
        gain_map = self.from_channels(x)
        noise = noise_map * gain_map
        x = x + noise
        return x 

class Blur(nn.Module):
    """Some Information about Blur"""
    def __init__(self):
        super(Blur, self).__init__()
        self.kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32)
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel[None, None, :, :]
    def forward(self, x):
        shape = x.shape
        # padding
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        # reshape
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        # convolution
        x = F.conv2d(x, self.kernel.to(x.device), stride=1, padding=0, groups=x.shape[1])
        # reshape
        x = x.reshape(shape)
        return x

class ToRGB(nn.Module):
    """Some Information about ToRGB"""
    def __init__(self,channels):
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()
    def forward(self, x):
        return self.tanh(self.conv(x))

class GeneratorBlock(nn.Module):
    """Some Information about GeneratorBlock"""
    def __init__(self, input_channels, latent_channels, output_channels, style_dim):
        super(GeneratorBlock, self).__init__()
        self.affine1 = nn.Linear(style_dim, latent_channels)
        self.conv1 = Conv2dMod(input_channels, latent_channels)
        self.bias1 = Bias(latent_channels)
        self.noise1 = NoiseInjection(latent_channels)
        self.activation1 = nn.LeakyReLU(0.2)
        
        self.affine2 = nn.Linear(style_dim, output_channels)
        self.conv2 = Conv2dMod(latent_channels, output_channels)
        self.noise2 = NoiseInjection(output_channels)
        self.bias2 = Bias(output_channels)
        self.activation1 = nn.LeakyReLU(0.2)
        
        self.to_rgb = ToRGB(output_channels)
    def forward(self, x, y):
        x = self.conv1(x, self.affine1(y))
        x = self.noise1(x)
        x = self.bias1(x)
        x = self.activation1(x)
        
        x = self.conv2(x, self.affine2(y))
        x = self.noise2(x)
        x = self.bias2(x)
        x = self.activation1(x)
        rgb = self.to_rgb(x)
        
        return x, rgb

class MappingNetwork(nn.Module):
    """Some Information about MappingNetwork"""
    def __init__(self, style_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()
        self.norm = nn.LayerNorm(style_dim)
        self.layers = nn.Sequential(*[nn.Linear(style_dim, style_dim) for _ in range(num_layers)])
    def forward(self, x):
        x = self.layers(self.norm(x))
        return x

class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self, initial_channels=512, style_dim=512):
        super(Generator, self).__init__()
        self.alpha = 0
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Blur(),
        )
        self.last_channels = initial_channels
        self.style_dim = style_dim
        self.layers = nn.ModuleList()
        self.const = nn.Parameter(torch.zeros(initial_channels, 4, 4, dtype=torch.float32))
        
        self.add_layer(initial_channels)
    def forward(self, style):
        x = self.const.repeat(style.shape[0], 1, 1, 1)
        alpha = self.alpha
        rgb_out = None
        num_layers = len(self.layers)
        if type(style) != list:
            style = [style] * num_layers
        
        for i in range(num_layers):
            x, rgb = self.layers[i](x, style[i])
            x = self.upscale(x)
            if i == num_layers - 1:
                rgb = rgb * alpha
                
            if rgb_out is None:
                rgb_out = rgb
            else:
                rgb_out = self.upscale(rgb_out) + rgb
        return rgb_out

    def add_layer(self, channels):
        self.layers.append(GeneratorBlock(self.last_channels, self.last_channels, channels, self.style_dim))
        self.last_channels = channels
        return self
        
class FromRGB(nn.Module):
    """Some Information about FromRGB"""
    def __init__(self, channels):
        super(FromRGB, self).__init__()
        self.conv = nn.Conv2d(3, channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv(x)
        return x

class DiscriminatorBlock(nn.Module):
    """Some Information about DiscriminatorBlock"""
    def __init__(self, input_channels, latent_channels, output_channels):
        super(DiscriminatorBlock, self).__init__()
        self.from_rgb = FromRGB(input_channels)
        self.conv1 = nn.Conv2d(input_channels, latent_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.activation1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(latent_channels, output_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.activation2 = nn.LeakyReLU(0.2)
        self.conv_ch = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, padding_mode='replicate')
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        return x
    
class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self, initial_channels = 512):
        super(Discriminator, self).__init__()
        self.alpha = 0
        self.layers = nn.ModuleList()
        self.fc1 = nn.Linear(4 * 4 * initial_channels + 1, initial_channels)
        self.fc2 = nn.Linear(initial_channels, 1)
        self.downscale = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.last_channels = initial_channels
        
        self.add_layer(initial_channels)
    def forward(self, rgb):
        num_layers = len(self.layers)
        alpha = self.alpha
        x = self.layers[0].from_rgb(rgb) * alpha
        for i in range(num_layers):
            x = self.layers[i](x) + self.layers[i].conv_ch(x)
            if i == 1:
                x += self.layers[i].from_rgb(self.downscale(rgb))
            if i < num_layers - 1:
                x = self.downscale(x)
        minibatch_std = torch.std(x, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(x.shape[0], 1)
        x = x.view(x.shape[0], -1)
        x = self.fc1(torch.cat([x, minibatch_std], dim=1))
        x = self.fc2(x)
        return x
    
    def add_layer(self, channels):
        self.layers.insert(0, DiscriminatorBlock(channels, channels, self.last_channels))
        self.last_channels = channels
        return self

class EMA(nn.Module):
    """Some Information about EMA"""
    def __init__(self, decay=0.9):
        super(EMA, self).__init__()
        self.decay = decay
        self.ema = None
    def forward(self, x):
        if self.ema is None:
            self.ema = x.detach()
        else:
            self.ema = self.ema * self.decay + x.detach() * (1 - self.decay)
        return x
        
class StyleGAN(nn.Module):
    """Some Information about StyleGAN"""
    def __init__(self, initial_channels = 512, style_dim = 512, min_channels = 16, max_resolution = 1024, initial_batch_size=32):
        super(StyleGAN, self).__init__()
        self.min_channels = min_channels
        self.max_resolution = max_resolution
        self.generator = Generator(initial_channels, style_dim)
        self.discriminator = Discriminator(initial_channels)
        self.mapping_network = MappingNetwork(style_dim)
        self.alpha = 0
        self.batch_size = initial_batch_size
        
    def train_resolution(self, dataset, batch_size, augment_func):
        dataloader = torch.utils.DataLoader(dataset, batch_size=batch_size, suffle=True)
    
    def hinge_loss_d(self, logit_real, logit_fake):
        loss_real = - torch.min(logit_real -1, torch.zeros_like(logit_real))
        loss_fake = - torch.min(-logit_fake -1, torch.zeros_like(logit_fake))
        
    def hinge_loss_g(self, logit_fake):
        return torch.mean(-logit_fake)
    