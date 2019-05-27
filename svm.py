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

clf = svm.SVC(gamma=0.1, C=0.01, kernel='poly', decision_function_shape='ovo')
print(cross_val_score(clf, train_images2[:40000], train_labels2[:40000], cv=2))
params = {
          'gamma': [1e-5,1e-4,1e-3],
          'C': [1e-3,1e-2,0.2]}
#gs = GridSearchCV(clf, params, refit=False, cv=2, scoring='accuracy')
#gs.fit(train_images2[:1000], train_labels2[:1000])
#print(gs.best_score_, gs.best_params_)
