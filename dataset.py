import torchvision.transforms.functional as TF
from torchvision import transforms
import os
from torch.utils.data import Dataset
import cv2
import random


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

class SameTransform(object):
    def __init__(self, mode, crop=None):
        self.np2tensor = transforms.ToTensor()
        self.mode = mode
        self.crop = crop
        self.lr_resize = transforms.Resize((120, 214), antialias = True)

    def __call__(self, lr, hr):
        lr = self.np2tensor(lr)
        hr = self.np2tensor(hr)

        if self.mode == 'train':
            lr, hr = self.same_transform(lr, hr)
            lr = self.lr_resize(lr)

        if self.crop:
            i, j, h, w = transforms.RandomCrop.get_params(lr, self.crop)
            lr = TF.crop(lr, i, j, h, w)
            hr = TF.crop(hr, i, j, h, w)
            
        return lr, hr#np.expand_dims(lr, 0), np.expand_dims(hr, 0)
    
    # после преобразований lr и hr сохраняют пространственное соотношение
    def same_transform(self, image1, image2, p=0.5):
        if random.random() > p:
            image1 = TF.hflip(image1)
            image2 = TF.hflip(image2)

        if random.random() > p:
            image1 = TF.vflip(image1)
            image2 = TF.vflip(image2)

        return image1, image2
    
    