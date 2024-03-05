import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_activation, **kwargs):
        super(ConvBlock, self).__init__()
        
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            **kwargs,
            bias=True,
        )
        
        self.activation = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()
    
    def forward(self, x):
        return self.activation(self.cnn(x))
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias = True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))
        

class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels = 32, residual_beta = 0.2):
        super(DenseResidualBlock, self).__init__()
        
        self.residual_beta = residual_beta
        self.subblocks = nn.ModuleList
        
        for i in range(5):
            self.subblocks.append(
                # channels * i for concatenation
                ConvBlock(
                    in_channels + channels * i,
                    channels if i < 4 else in_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1,
                    use_activation = (i < 4)
                )
            )
    
    def forward(self, x):
        new_input = x
        for block in self.subblocks:
            out = block(new_input)
            new_input = torch.cat([new_input, out], dim=1)
        return self.residual_beta*out + x
    
class RDRB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super(RDRB, self).__init__()
        self.residual_beta = residual_beta
        self.rdrb = nn.Sequential(
            *[DenseResidualBlock(in_channels) for _ in range(3)]
        )
    
    def forward(self, x):
        return self.rdrb(x) + self.residual_beta * x

class Generator(nn.Module):
    def __init__(self, in_channels, num_channels = 64, num_blocks = 23) -> None:
        super(Generator, self).__init__()
        self.initial = nn.Conv2d(
            in_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        
        self.residuals = nn.Sequential(
            *[RDRB(num_channels) for _ in range(num_blocks)]
        )
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.upsampling = nn.Sequential(
            UpsampleBlock(num_channels), UpsampleBlock(num_channels)
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, 3, 1, 1, bias=True)
        )
        
    def forward(self, x):
        initial = self.initial(x)
        x = self.conv(self.residuals(initial)) + initial
        x = self.upsampling(x)
        return self.final(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, features = [64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size = 3,
                    stride = 1 + idx % 2,
                    padding = 1,
                    use_activation=True
                )
            )
            in_channels = feature
        
        self.blocks = nn.Sequential(*blocks)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)
        
# def initialize_weights(model, scale = 0.1):
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal(m.weight.data)
#             m.weight.data *= scale
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_normal(m.weight.data)
#             m.weight.data *= scale
#         # elif isinstance(m. nn.Module):
#         #     initialize_weights(m)
        
# to train -> 1000000 with only nn.L1Loss, then we introduce discriminator
