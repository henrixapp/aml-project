import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from tqdm import tqdm
#import matplotlib


class InfektaDataset(Dataset):
    def __init__(self, img_dir,time_frames_count, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.folders = glob.glob(img_dir+"/*")
        print("Number of runs:",len(self.folders))
        self.folders_count = len(self.folders)
        self.time_frames_count = time_frames_count
        self.sample_per_folder = len(glob.glob(self.folders[0]+"/*"))-time_frames_count-1
        self.people_count = np.sum(np.load(glob.glob(self.folders[0]+"/*")[0]))
        self.people_factor = 1.0/self.people_count
        print(self.people_count)
        print(self.sample_per_folder)
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        for folder in tqdm(self.folders):
            fdata = []
            for i in range(len(glob.glob(self.folders[0]+"/*"))):
                img_path = os.path.join(folder, str(i)+".npy")
                fdata += [np.load(img_path)*self.people_factor]
            self.data += [np.vstack(fdata)]



    def __len__(self):
        return self.folders_count*self.sample_per_folder
    def __getitem__(self, idxn):
        idx = idxn % self.sample_per_folder
        idc = int(idxn / self.sample_per_folder)
        imgs = self.data[idc][idx*8:(idx+self.time_frames_count)*8]
        # for  i in range (self.time_frames_count):
        #     img_path = os.path.join(self.folders[idc], str(idx+i)+".npy")
        #     image = np.load(img_path)*self.people_factor
        #     imgs +=[image]
        # img_path = os.path.join(self.folders[idc], str(idx+self.time_frames_count)+".npy")
        # target = np.load(img_path)*self.people_factor
        target = self.data[idc][(idx+self.time_frames_count)*8:(idx+self.time_frames_count+1)*8]
        if self.transform:
            image = self.transform(imgs)
        if self.target_transform:
            target = self.target_transform(target)
        return imgs, target
if __name__ =="__main__":
    infekta = InfektaDataset("../INFEKTA-HD/data/runs16/",2)
    print(len(infekta))

    #matplotlib.image.imsave(str(0).zfill(5) + ".png", infekta[7][0][3])