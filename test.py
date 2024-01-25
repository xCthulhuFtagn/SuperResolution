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


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        # feature extraction
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        # shrinking
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        # mapping
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        # expanding
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        # Deconvolution
        # originally found d instead of s in picture
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


class SRDataset(Dataset):
    def __init__(self, lr_path, hr_path, transform = None):
        self.lr = [os.path.join(lr_path, f) for f in os.listdir(lr_path)]
        self.hr = [os.path.join(hr_path, f) for f in os.listdir(hr_path)]
        self.lr, self.hr = sorted(self.lr), sorted(self.hr)
        assert len(self.lr) == len(self.hr)
        self.transform = transform

    def __len__(self):
        return len(self.lr)

    def file2np(self, path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        lr = self.file2np(self.lr[idx])
        hr = self.file2np(self.hr[idx])
        if self.transform is not None: lr, hr = self.transform(lr, hr)
        return lr, hr


