import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import ImageDataset
from model import StyleGAN


if os.path.exists("model.pt"):
    model = torch.load("model.pt")
    print("Loaded model from disk")
else:
    model = StyleGAN()
    print("Created new model")
dataset = ImageDataset(source_dir_pathes=sys.argv[1:], chache_dir="./dataset_chache/", max_len=500)
model.train(dataset, batch_size=32, num_epoch=30)
