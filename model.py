import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import multiprocessing
import time

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
        self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float32))

    def forward(self, x):
        # x: (batch_size, channels, H, W)
        # self.noise: (channels)
        
        # add bias
        bias = self.bias.repeat(x.shape[0], 1, x.shape[2], x.shape[3]).reshape(x.shape)
        x = x + bias
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
    def forward(self, x):
        x = self.conv(x)
        return x
    
class NoiseInjection(nn.Module):
    """Some Information about NoiseInjection"""
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        noise = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], dtype=torch.float32).to(x.device)
        noise = self.conv(noise)
        x = x + noise
        return x

class EqualLinear(nn.Module):
    """Some Information about EqualLinear"""
    def __init__(self, input_dim, output_dim):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
class MappingNetwork(nn.Module):
    """Some Information about MappingNetwork"""
    def __init__(self, latent_dim, num_layers=8):
        super(MappingNetwork, self).__init__()
        self.seq = nn.Sequential(*[nn.Sequential(EqualLinear(latent_dim, latent_dim), nn.LayerNorm(latent_dim)) for _ in range(num_layers)])
    def forward(self, x):
        return self.seq(x)
    
class GeneratorBlock(nn.Module):
    """Some Information about GeneratorBlock"""
    def __init__(self, input_channels, latent_channels, output_channels, style_dim, upsample=True):
        super(GeneratorBlock, self).__init__()
        if upsample:
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), Blur())
        else:
            self.upsample = nn.Identity()
        self.affine1 = nn.Linear(style_dim, latent_channels)
        self.conv1dw = nn.Conv2d(input_channels, input_channels, 3, 1, 1, groups=input_channels, padding_mode='replicate')
        self.conv1cw = Conv2dMod(input_channels, latent_channels, kernel_size=1)
        #self.noise1 = NoiseInjection(latent_channels)
        self.bias1 = Bias(latent_channels)
        self.activation1 = nn.LeakyReLU(0.2)
        
        self.affine2 = nn.Linear(style_dim, output_channels)
        self.conv2dw = nn.Conv2d(latent_channels, latent_channels, 3, 1, 1, groups=latent_channels, padding_mode='replicate')
        self.conv2cw = Conv2dMod(latent_channels, output_channels, kernel_size=1)
        #self.noise2 = NoiseInjection(output_channels)
        self.bias2 = Bias(output_channels)
        self.activation2 = nn.LeakyReLU(0.2)
        
        self.to_rgb = ToRGB(output_channels)
    def forward(self, x, y):
        x = self.conv1dw(x)
        x = self.conv1cw(x, self.affine1(y))
        #x = self.noise1(x)
        x = self.bias1(x)
        x = self.activation1(x)
        
        x = self.conv2dw(x)
        x = self.conv2cw(x, self.affine2(y))
        #x = self.noise2(x)
        x = self.bias2(x)
        x = self.activation2(x)
        rgb = self.to_rgb(x)
        return x, rgb

class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self, initial_channels=512, style_dim=512):
        super(Generator, self).__init__()
        self.alpha = 0
        self.upscale = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), Blur())
        self.last_channels = initial_channels
        self.style_dim = style_dim
        self.layers = nn.ModuleList()
        self.const = nn.Parameter(torch.randn(initial_channels, 4, 4, dtype=torch.float32))
        self.tanh = nn.Tanh()
        self.add_layer(initial_channels, upsample=False)
        
    def forward(self, style):
        x = self.const.repeat(style.shape[0], 1, 1, 1)
        alpha = self.alpha
        rgb_out = None
        num_layers = len(self.layers)
        if type(style) != list:
            style = [style] * num_layers
        
        for i in range(num_layers):
            x = self.layers[i].upsample(x)
            x, rgb = self.layers[i](x, style[i])
            if i == num_layers-1:
                rgb = rgb * alpha
            if rgb_out == None:
                rgb_out = rgb
            else:
                rgb_out = self.upscale(rgb_out) + rgb
        rgb_out = self.tanh(rgb_out)
        return rgb_out

    def add_layer(self, channels, upsample=True):
        self.layers.append(GeneratorBlock(self.last_channels, self.last_channels, channels, self.style_dim, upsample=upsample))
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

class Conv2dXception(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'):
        super(Conv2dXception, self).__init__()
        self.depthwise   = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, padding_mode=padding_mode, groups=input_channels)
        self.channelwise = nn.Conv2d(input_channels, output_channels, 1, 1, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.channelwise(x)
        return x

class DiscriminatorBlock(nn.Module):
    """Some Information about DiscriminatorBlock"""
    def __init__(self, input_channels, latent_channels, output_channels):
        super(DiscriminatorBlock, self).__init__()
        self.from_rgb = FromRGB(input_channels)
        self.conv1 = Conv2dXception(input_channels, latent_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.activation1 = nn.LeakyReLU(0.2)
        self.conv2 = Conv2dXception(latent_channels, output_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.activation2 = nn.LeakyReLU(0.2)
        self.conv_ch = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, padding_mode='replicate')
        
    def forward(self, x):
        res = self.conv_ch(x)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = x + res
        return x
    
class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self, initial_channels = 512):
        super(Discriminator, self).__init__()
        self.alpha = 0
        self.layers = nn.ModuleList()
        self.fc1 = nn.Linear(initial_channels + 2, 512)
        self.activation1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.last_channels = initial_channels
        self.blur = Blur()
        
        self.add_layer(initial_channels)

    def forward(self, rgb):
        num_layers = len(self.layers)
        alpha = self.alpha
        x = self.layers[0].from_rgb(rgb)
        color_minibatch_std = torch.std(x, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(x.shape[0], 1)
        for i in range(num_layers):
            if i == 1:
                x = x * alpha + self.layers[1].from_rgb(self.pool(self.blur(rgb))) * (1-alpha)
            x = self.layers[i](x)
            if i < num_layers - 1:
                x = self.pool(x)
        minibatch_std = torch.std(x, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(x.shape[0], 1)
        x = self.pool(self.pool(x))
        x = x.view(x.shape[0], -1)
        x = self.fc1(torch.cat([x, minibatch_std, color_minibatch_std], dim=1))
        x = self.activation1(x)
        x = self.fc2(x)
        return x
    
    def add_layer(self, channels):
        self.layers.insert(0, DiscriminatorBlock(channels, self.last_channels, self.last_channels))
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
        return self.ema
        
class StyleGAN(nn.Module):
    """Some Information about StyleGAN"""
    def __init__(self, initial_channels = 512, style_dim = 512, min_channels = 16, max_resolution = 1024, initial_batch_size=32):
        super(StyleGAN, self).__init__()
        self.min_channels = min_channels
        self.max_resolution = max_resolution
        self.generator = Generator(initial_channels, style_dim)
        self.initial_channels = initial_channels
        self.style_dim = style_dim
        self.discriminator = Discriminator(initial_channels)
        self.mapping_network = MappingNetwork(style_dim)
        self.batch_size = initial_batch_size
        
    def train(self, dataset, batch_size,  *args, **kwargs):
        image_size = 4
        while image_size < self.max_resolution:
            # get number of layers
            num_layers = len(self.discriminator.layers)
            image_size = 4 * 2 ** (num_layers - 1)
            if image_size > self.max_resolution:
                break
            bs = batch_size // (2 ** (num_layers - 1))
            if bs < 4:
                bs = 4
            bs = int(bs)
            # get number of channels
            dataset.set_size(image_size)
            self.train_resolution(dataset, bs, *args, **kwargs)
            
            channels = self.initial_channels // (2 ** (num_layers-1))
            channels = int(channels)
            if channels < 8:
                channels = 8
            
            print(f"batch size: {bs}, channels: {channels}")
            self.generator.add_layer(channels)
            self.discriminator.add_layer(channels)
        
    def train_resolution(self, dataset, batch_size, augment_func=nn.Identity(), num_epoch=1, model_path='model.pt', result_dir_path='results', smooth_growning=False, distangle=False):
        if not os.path.exists(result_dir_path):
            os.mkdir(result_dir_path)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        opt_d, opt_g, opt_m = torch.optim.Adam(self.discriminator.parameters(), lr=1e-5), torch.optim.Adam(self.generator.parameters(), lr=1e-5), torch.optim.Adam(self.mapping_network.parameters(), lr=1e-5)
        D, G, M = self.discriminator, self.generator, self.mapping_network
        D.alpha = 1 / num_epoch
        G.alpha = 1 / num_epoch
        
        D.to(device)
        G.to(device)
        M.to(device)
        ema = EMA(0.999)

        bar = tqdm(total=num_epoch * (int(len(dataset) / batch_size) + 1), position=1)
        
        for epoch in range(num_epoch):
            for i, image in enumerate(dataloader):
                # Train generator
                if len(D.layers) == 1 or (not smooth_growning):
                    G.alpha = 1
                    D.alpha = 1

                M.zero_grad()
                G.zero_grad()
                z = torch.randn(image.shape[0], self.style_dim).to(device)
                w = M(z)
                N = image.shape[0]
                fake_image = G(w)
                generator_adversarial_loss = -D(fake_image).mean()
                generator_loss = generator_adversarial_loss

                if i % 16 == 0 and epoch > num_epoch // 4 and distangle:
                    G.zero_grad()
                    fi = fake_image
                    #w.requires_grad = True
                    #fi.requires_grad = True
                    #print(w)
                    dw = (w[1] - w[0]).reshape(1, -1) # 1, N
                    dgw = (fi[1] - fi[0]).reshape(-1, 1) # M, 1
                    #print(dw[0, 0], dgw[0])
                    Jw = torch.mm(dgw, 1/dw)
                    #print(Jw.shape)
                    l2n = torch.sqrt((Jw.reshape(-1) ** 2).sum())
                    a = ema(l2n)
                    err = (l2n - a) ** 2
                    generator_loss += err
                    tqdm.write(f"Smooth loss: {err}, L2: {l2n}")

                generator_loss.backward()
                
                opt_g.step()
                opt_m.step()

                # Train discriminator
                D.zero_grad()
                real_image = augment_func(image.to(device))
                fake_image = augment_func(fake_image.detach())
                discriminator_loss_real = -torch.minimum(D(real_image)-1, torch.zeros(N, 1).to(device)).mean()
                discriminator_loss_fake = -torch.minimum(-D(fake_image)-1, torch.zeros(N, 1).to(device)).mean()
                discriminator_loss = (discriminator_loss_real + discriminator_loss_fake)
                discriminator_loss.backward()
                
                # update parameters
                opt_d.step()
                
                # update progress bar
                bar.set_description(f"Epoch: {epoch} Batch: {i} DLoss: {discriminator_loss.item():.4f}, GLoss: {generator_loss.item():.4f}, alpha: {G.alpha:.4f}")
                if i % 100 == 0:
                    tqdm.write(f"DLosses: Fake:{discriminator_loss_fake:.4f}, Real: {discriminator_loss_real:.4f}")
                    tqdm.write(f"GLosses: Adversarial: {generator_adversarial_loss:.4f}")
                bar.update(1)
                D.alpha = epoch / num_epoch + (i / (int(len(dataset) / batch_size) + 1)) / num_epoch
                G.alpha = epoch / num_epoch + (i / (int(len(dataset) / batch_size) + 1)) / num_epoch
                
            torch.save(self, model_path)
            # write image
            image = fake_image[0].detach().cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = image * 127.5 + 127.5
            image = image.astype(np.uint8)
            image = Image.fromarray(image, mode='RGB')
            image.save(os.path.join(result_dir_path, f"{epoch}.png"))
    
    def generate_random_image(self, num_images):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images = []
        for i in range(num_images):
            style = torch.randn(1, self.style_dim).to(device)
            style = self.mapping_network(style)
            image = self.generator(style)
            image = image.detach().cpu().numpy()
            images.append(image[0])
        return images
    
    def generate_random_image_to_directory(self, num_images, dir_path="./tests"):
        images = self.generate_random_image(num_images)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for i in range(num_images):
            image = images[i]
            image = np.transpose(image, (1, 2, 0))
            image = image * 127.5 + 127.5
            image = image.astype(np.uint8)
            image = Image.fromarray(image, mode='RGB')
            image = image.resize((1024, 1024))
            image.save(os.path.join(dir_path, f"{i}.png"))
            
    def generate_gif(self, num_images, output_path="output.gif"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        style1, style2 = torch.randn(1, self.style_dim).to(device), torch.randn(1, self.style_dim).to(device)
        M, G = self.mapping_network, self.generator
        style1, style2 = M(style1), M(style2)
        images = []
        with torch.no_grad():
            for i in range(num_images):
                alpha = i / num_images
                style = style1 * alpha + style2 * (1 - alpha)
                image = G(style)
                image = image.detach().cpu().numpy()[0]
                image = np.transpose(image, (1, 2, 0))
                image = image * 127.5 + 127.5
                image = image.astype(np.uint8)
                images.append(image)
        # save gif
        images = [Image.fromarray(image, mode='RGB') for image in images]
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=100, loop=0)
