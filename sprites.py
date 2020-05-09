import os
import numpy as np
import time
import pickle
from torch.utils.data import Dataset

class Sprites(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.path = './splits/'
        self.files = os.listdir(self.path+self.split)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        with open(self.path+self.split+"/"+file,'rb') as f:
            dict_pair = pickle.load(f)
        dict_pair['img1'] = dict_pair['img1'][:,:,:3]
        dict_pair['img2'] = dict_pair['img2'][:,:,:3]
        return dict_pair
