# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:03:44 2019

@author: Alexandre
"""

import os
import matplotlib.pyplot as plt


directory  = os.getcwd()

def printer(img,label):
    plt.figure()
    plt.title('Label: {}'.format(label))
    plt.imshow(img)
    
    
    
def vect(Choix):
    if Choix == 0:
        Y = [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    elif Choix == 1:
        Y = [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.]
    elif Choix == 2:
        Y = [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.]
    elif Choix == 3:
        Y = [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]
    elif Choix == 4:
        Y = [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.]
    elif Choix == 5:
        Y = [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.]
    elif Choix == 6:
        Y = [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]
    elif Choix == 7:
        Y = [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.]
    elif Choix == 8:
        Y = [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]
    elif Choix == 9:
        Y = [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]
    return Y



def Accuracy(y_pred,train_labels):
    N = len(train_labels)
    acc = 0.
    for ide in range(N):
        a = [y_pred[ide,i].item() for i in range(10)]
        if a.index(max(a)) == train_labels.iloc[ide]['Category']:
            acc += 1.
    acc = acc/N
    return acc



def Accuracy_debuguage(y_pred,labels,images,data_set):
    N = len(labels)
    acc = 0.
    for ide in range(N):
        a = [y_pred[ide,i].item() for i in range(10)]
        if a.index(max(a)) == labels.iloc[ide]['Category']:
            acc += 1.
        else:
            plt.figure()
            plt.title('Label: {0} -> {1}'.format(labels.iloc[ide]['Category'],a.index(max(a))))
            plt.imshow(images[ide])
            plt.savefig(directory+"/results/BadPredictions/{0}/img_{1}.png".format(data_set,ide))
            plt.close()
    acc = acc/N
    return acc
