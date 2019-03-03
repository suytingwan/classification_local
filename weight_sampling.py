from collections import defaultdict
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import time
import torch

def default_loader(path):
    try:
        image = Image.open(path).convert('RGB')
    except Exception as e:
        print(e)
        return None
    
    return image


def default_flist_reader(root, flist):
    """
    flist format: img_path label\nimg_path label\n ...(same to caffe's filelist)
    """
    img_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            img_path, img_label, img_prob = line.strip().split()
            if os.path.isfile(os.path.join(root, img_path)):
                img_list.append((img_path, int(img_label), float(img_prob)))
                    
    return img_list

def weighted_pick(weights):
    cum_sum = np.cumsum(weights)
    sum = np.sum(weights)
    pick_id = int(np.searchsorted(cum_sum, np.random.rand(1)*sum)) #binary search 
    return  pick_id


class ImageWeightSampling(data.Dataset):
    """A generic data loader where the image paths are in file

    """
    def __init__(self, root, flist, transform=None, target_transform=None,
            flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.img_list = flist_reader(self.root, flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        self.sample_list = defaultdict(list)
        self.prob_list = defaultdict(list)
        
        for index,  (img_path, pid, prob) in enumerate(self.img_list):
            self.sample_list[pid].append(img_path)
            self.prob_list[pid].append(prob)
        
        self.pids = list(self.sample_list.keys())
        self.pids_num = len(self.pids)

    def __getitem__(self, index):
        index = index % self.pids_num 
        target = self.pids[index]
        imgs = self.sample_list[target]
        prob = self.prob_list[target]
        
        pick_id = weighted_pick(prob) 
        img_path = imgs[pick_id] 
        img_path = os.path.join(self.root, img_path)
        img = self.loader(img_path)
        
        #print('img_path:%s\tlabel:%s' % (img_path, target))
        
        while True:
            if img is not None:
                break
            else:
                img_path = np.random.choice(imgs, 1, replace=False)
                img_path = os.path.join(self.root, img_path)
                img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        #return len(self.pids)
        return len(self.img_list)


if __name__ == '__main__':
    root = '/home/ssd5/vis/data/webvision' 
    flist = '/home/disk1/data/test/weight_samples.txt'
    #ImageFileList('', flist)
    ImageWeightSampling(root, flist)
