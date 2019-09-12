'''
Created on Sep 10, 2019

@author: monky
'''
'''
Created on Sep 9, 2019

@author: mitom
'''

import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
import time
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
import torchvision as tv

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from box_dataset import BoxDataset



class BBRegression(nn.Module):
    '''
    classdocs
    '''

    def __init__(self):
        super(BBRegression, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 16, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(),  
                                   nn.MaxPool2d((2,2), stride=(2,2)))
        self.block2 = nn.Sequential(nn.Conv2d(16, 32, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(),
                                   nn.MaxPool2d((2,2), stride=(2,2)))
        self.classifier = nn.Sequential(
            nn.Linear(25*25*32, 100),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(100, 4),
        )
    
    def forward(self, x):
        x = self.block1(x) 
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
    
def comput_iou(box1, box2):
    
    
    xmin1, ymin1 = int(box1[0]), int(box1[1])
    xmax1, ymax1 = int(box1[0]+box1[2]), int(box1[1]+box1[3])
    xmin2, ymin2 = int(box2[0]), int(box2[1])
    xmax2, ymax2 = int(box2[0]+box2[2]), int(box2[1]+box2[3])
    
    # get lefttop and rightdown
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    
    area1 = (xmax1-xmin1) * (ymax1-ymin1) 
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))
    iou = inter_area / (area1+area2-inter_area+1e-6)
  
    
    return xx1, xx2, yy1, yy2, iou
    
    
def generate_img(list_t,list_p):
    img = np.zeros([100, 100])

    x_offset_t = list_t[0]
    y_offset_t = list_t[1]
    curr_w_t = list_t[2]
    curr_h_t = list_t[3]
    
    
    x_offset_p = list_p[0]
    y_offset_p = list_p[1]
    curr_w_p = list_p[2]
    curr_h_p = list_p[3]
    
    
    img[y_offset_t:y_offset_t + curr_h_t, x_offset_t:x_offset_t + curr_w_t] = 100
    img[y_offset_p:y_offset_p + curr_h_p, x_offset_p:x_offset_p + curr_w_p] = 200
    
    xx1, xx2, yy1, yy2, iou = comput_iou(list_t, list_p)
    iou = ("%.4f" % iou)
    img[yy1:yy2,xx1:xx2] = 150
    
    
    label = [x_offset_t, y_offset_t, curr_w_t, curr_h_t]
    
    
    
    plt.title('target:   x:{}, y:{}, w:{}, h:{}, iou:{}'.format(list_t[0], list_t[1], list_t[2], list_t[3], iou))
    plt.xlabel('preloc:   x:{}, y:{}, w:{}, h:{}'.format(list_p[0], list_p[1], list_p[2], list_p[3]))
    plt.ioff()
    plt.imshow(img,  cmap='Greys_r')
    plt.pause(0.1)
        #plt.pause(1)   
    
    
    
model = BBRegression()

root = "/home/mitom/dlcv/data/boundingbox/"
datatxt = "box_label.txt"
# datatxt_test = "test_labels.txt"
train_data= BoxDataset(root,datatxt, transform=tv.transforms.ToTensor())
train_loader = data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_data= BoxDataset(root,datatxt, transform=tv.transforms.ToTensor())
test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.MSELoss()
for epoch in range(10):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        targets = targets.float()
        
        loss = loss_func(logits, targets)
        
        #loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        loss.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        plt.ion()
        ### LOGGING
        if not batch_idx % 2:
            print('---------------------------------------')
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.8f' 
                   %(epoch+1, 10, batch_idx, 
                     len(train_loader), loss))
            
            model.eval()
                
            for i, (features, targets) in enumerate(test_loader):
                logits, probas = model(features)
                
                print('target :   ', targets)
                print('predict:   ', logits)
                targets = targets.detach().numpy().tolist()
                logits = logits.detach().numpy().tolist()
                targets = [int(i*100) for i in targets[0]]
                logits = [int(i*100) for i in logits[0]]
                
                generate_img(targets, logits)
                break
            

    

    
    
    
    
    
    