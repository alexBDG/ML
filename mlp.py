import sys
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skorch import * 
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import cross_val_score 
import modules

train_images = pd.read_pickle('input/new_train_images.pkl')
train_labels = pd.read_csv('input/train_labels.csv')

train_images2 = train_images.reshape(len(train_images), 51*51)
train_labels2 = np.zeros(len(train_labels), dtype=int)
for i in range(len(train_labels)):
  train_labels2[i] = train_labels.iloc[i]['Category']

net = NeuralNetClassifier(
        modules.MLP,
        criterion = torch.nn.CrossEntropyLoss,
        max_epochs=10,
        batch_size = 20,
        lr=1e-2,
        )
#net.fit(train_images2, train_labels2)
params = {
          'lr': [1e-2, 1e-4, 1e-6],
          'module__num_units1':[500,600],
          'module__num_units2':[100,200],
          }
gs = GridSearchCV(net, params, refit=False, cv=2, scoring='accuracy')
gs.fit(train_images2[:1000], train_labels2[:1000])
print(gs.best_score_, gs.best_params_)
