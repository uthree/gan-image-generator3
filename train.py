import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import ImageDataset
from model import StyleGAN

model = StyleGAN()
dataset = ImageDataset(source_dir_pathes=sys.argv[1:], chache_dir="./dataset_chache/", max_len=500)
model.train(dataset, batch_size=32, num_epoch=50)
