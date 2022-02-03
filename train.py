import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import ImageDataset
from model import StyleGAN
from augmentations import flip, contrast, random_roll
from augmentor import ImageDataAugmentor
aug = ImageDataAugmentor()
aug.add_function(flip)
aug.add_function(contrast)
aug.add_function(random_roll)

if os.path.exists("model.pt"):
    model = torch.load("model.pt")
    print("Loaded model from disk")
else:
    model = StyleGAN(max_resolution=1024, initial_channels=512)
    print("Created new model")
dataset = ImageDataset(source_dir_pathes=sys.argv[1:], chache_dir="./dataset_chache/", max_len=100)
model.train(dataset, batch_size=64, num_epoch=30,  augment_func=aug)
