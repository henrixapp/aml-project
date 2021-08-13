import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
#import matplotlib


class InfektaDataset(Dataset):
    def __init__(self, img_dir,transform=None, target_transform=None):
        self.img_dir = img_dir
        self.files = glob.glob(img_dir+"/*")
        print(len(self.files))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)-1

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(idx)+".npy")
        img_path1 = os.path.join(self.img_dir, str(idx+1)+".npy")

        image = np.load(img_path)
        target = np.load(img_path1)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target
if __name__ =="__main__":
    infekta = InfektaDataset("../INFEKTA-HD/dump/")
    print(infekta[0])
    #matplotlib.image.imsave(str(0).zfill(5) + ".png", infekta[7][0][3])