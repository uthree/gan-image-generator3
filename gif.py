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

model.generate_gif(int(sys.argv[1]))