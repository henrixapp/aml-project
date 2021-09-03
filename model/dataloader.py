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
        self.folders = glob.glob(img_dir+"/*")
        print(self.folders)
        self.folders_count = len(self.folders)
        self.sample_per_folder = len(glob.glob(self.folders[0]+"/*"))-5
        self.people_count = np.sum(np.load(glob.glob(self.folders[0]+"/*")[0]))
        self.people_factor = 1.0/self.people_count
        print(self.people_count)
        print(self.sample_per_folder)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.folders_count*self.sample_per_folder
    def __getitem__(self, idxn):
        idx = idxn % self.sample_per_folder
        idc = int(idxn / self.sample_per_folder)
        img_path = os.path.join(self.folders[idc], str(idx)+".npy")
        img_path1 = os.path.join(self.folders[idc], str(idx+1)+".npy")
        img_path2 = os.path.join(self.folders[idc], str(idx+2)+".npy")
        img_path3 = os.path.join(self.folders[idc], str(idx+3)+".npy")
        img_path4 = os.path.join(self.folders[idc], str(idx+4)+".npy")

        image = np.load(img_path)*self.people_factor
        image1 = np.load(img_path1)*self.people_factor
        image2 = np.load(img_path2)*self.people_factor
        image3 = np.load(img_path3)*self.people_factor
        target = np.load(img_path4)*self.people_factor
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return np.vstack([image,image1,image2,image3]), target
if __name__ =="__main__":
    infekta = InfektaDataset("../INFEKTA-HD/dump/")
    print(infekta[0])
    #matplotlib.image.imsave(str(0).zfill(5) + ".png", infekta[7][0][3])