import torch
import torchvision
from torchvision import transforms, utils
import pandas as pd
import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
import random

class NoisyDataset(Dataset):

    def __init__(self, root, filename, transform=None, mix=False, plus=False):

        self.ids = []
        self.targets = []
        self.weights = []
        self.sample_weights = []

        f = open(filename)
        for line in f:
            info = line.strip().split(' ')
            self.ids.append(info[0])
            if len(info) == 1:
                self.targets.append(np.array([0],dtype=np.int32))
            else:
                self.targets.append(np.array([info[1]], dtype=np.int32))
            if len(info) > 2:
                self.weights.append(np.array([info[2]], dtype=np.float32))
                self.sample_weights.append(float(info[2]))
            else:
	            self.weights.append(np.array([1.0], dtype=np.float32))

        self.root_dir = root
        self.transform = transform
        self.mix = mix
        self.plus = plus

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.mix:
            while True:
                index1 = random.randint(0, len(self.ids) - 1)
                index2 = random.randint(0, len(self.ids) - 1)

                label1 = self.targets[index1]
                label2 = self.targets[index2]
                if label1 != label2:
                    break

            image_name1 = os.path.join(self.root_dir, self.ids[index1])
            image_name2 = os.path.join(self.root_dir, self.ids[index2])

            image1 = Image.open(image_name1).convert('RGB')
            image2 = Image.open(image_name2).convert('RGB')

            r = np.array(random.random())
            if self.plus:
                g1 = np.std(image1)
                g2 = np.std(image2)
                p = 1.0 / (1 + g1 / g2 * (1-r) / r)
                image = ((image1) * p + image2 * (1-p))/ np.sqrt(p ** 2 + (1-p) ** 2).astype(np.float32)
            else:
                image = (image1 * r + image2 * (1-r)).astype(np.float32)

            if self.transform:
                image = self.transform(image)

            landmarks1 = torch.from_numpy(label1)
            landmarks2 = torch.from_numpy(label2)

            return image,landmarks1.long(), landmarks2.long(), r


        else:
            img_name = os.path.join(self.root_dir,self.ids[idx])
            image = Image.open(img_name).convert('RGB')
            landmarks = torch.from_numpy(self.targets[idx])
            weight = torch.from_numpy(self.weights[idx])

            #print(image)
            if self.transform:
                image = self.transform(image)
            #print(image)
            return image, landmarks.long(), weight
    
    def get_ids(self):
        return self.ids

    def get_weights(self):
        return self.sample_weights
