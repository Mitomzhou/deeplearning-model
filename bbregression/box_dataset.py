'''
Created on Sep 9, 2019

@author: monky
'''
#-*- coding:utf-8 -*-
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.optim
import torch.utils.data as data
import torch
import torchvision as tv
from PIL import Image
import numpy



class BoxDataset(data.Dataset):
    def __init__(self,root, datatxt, transform=None, target_transform=None):
        fh = open(root + datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split(" ")
            
            labellist = [float(words[1]),float(words[2]),float(words[3]),float(words[4])]
            imgs.append((words[0],labellist))
            self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)  
        label = torch.from_numpy(numpy.array(label))
        return img,label
   
    def __len__(self):  
        return len(self.imgs)
    
    
if __name__ == '__main__':
    root = "../../data/boundingbox/"
    datatxt = "box_label.txt"
    train_data=BoxDataset(root,datatxt, transform=tv.transforms.ToTensor())
    #test_data=MyDataset(txt=root+'test.txt', transform=tv.transforms.ToTensor())
    train_loader = data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    print("done")
    for batch_idx, (data, target) in enumerate(train_loader):
        print('---------------------------------------')
        print(batch_idx)
        print(data)
        print(target)
    
    
    
