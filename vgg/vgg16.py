'''
Created on Sep 3, 2019

@author: mitom
'''

import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
import time
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


"""
VGG16 A is not suitable for training cifar-10
"""
class VGG16(nn.Module):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(VGG16, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(),  
                                   nn.Conv2d(64, 64, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(),  
                                   nn.MaxPool2d((2,2), stride=(2,2)))
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(),
                                   nn.Conv2d(128, 128, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(), 
                                   nn.MaxPool2d((2,2), stride=(2,2)))
        self.block3 = nn.Sequential(nn.Conv2d(128, 256, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(),
                                   nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(), 
                                   nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(),
                                   nn.MaxPool2d((2,2), stride=(2,2)))
        self.block4 = nn.Sequential(nn.Conv2d(256, 512, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(),
                                   nn.Conv2d(512, 512, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(), 
                                   nn.Conv2d(512, 512, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(), 
                                   nn.MaxPool2d((2,2), stride=(2,2)))
        self.block5 = nn.Sequential(nn.Conv2d(512, 512, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(), 
                                   nn.Conv2d(512, 512, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(), 
                                   nn.Conv2d(512, 512, (3,3), stride=(1,1), padding=(1,1)), nn.ReLU(), 
                                   nn.MaxPool2d((2,2), stride=(2,2)))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
            
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()
    
    def forward(self, x):
        x = self.block1(x) 
        x = self.block2(x)
        x = self.block3(x) 
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 64

# Architecture
num_features = 784
num_classes = 10
train_dataset = datasets.CIFAR10(root='../../data/cifar-10/', 
                                 train=True, 
                                 transform=transforms.ToTensor(),
                                 download=True)

test_dataset = datasets.CIFAR10(root='../../data/cifar-10/', 
                                train=False, 
                                transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break




torch.manual_seed(random_seed)
model = VGG16()
#print(model)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
    
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


start_time = time.time()
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%% | Loss: %.3f%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader),
              compute_epoch_loss(model, train_loader)))


    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    

