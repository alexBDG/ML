# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:11:41 2019

@author: Alexandre
"""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
  def __init__(self,num_units1=250,num_units2=100):
    super(MLP, self).__init__()
    self.linear1 = nn.Linear(51*51,num_units1)
    self.linear2 = nn.Linear(num_units1,num_units2)
    self.linear3 = nn.Linear(num_units2,10)
                                        
  def forward(self,X):
    X = F.relu(self.linear1(X))
    X = F.relu(self.linear2(X))
    X = self.linear3(X)
    return F.log_softmax(X, dim=1)

class Net_1(nn.Module):
    
    def __init__(self):
        super(Net_1, self).__init__()
        self.lin1 = nn.Linear(51*51, 1048)
        self.lin2 = nn.Linear(1048, 10)
        
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    
    
    
class Net_2(nn.Module):
    
    def __init__(self):
        super(Net_2, self).__init__()
        self.lin1 = nn.Linear(51*51, 1048)
        self.lin2 = nn.Linear(1048, 64)
        self.lin3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
    
 
class Net_3(nn.Module):
    
    def __init__(self):
        super(Net_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(p=0.25)
        self.lin1 = nn.Linear(64*23*23,128)
        self.drop2 = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop1(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.lin1(x))
        x = self.drop2(x)
        x = F.softmax(self.lin2(x))
        return x

    
class Net_4(nn.Module):
    
    def __init__(self):
        super(Net_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.lin1 = nn.Linear(16*9*9, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
    
    
