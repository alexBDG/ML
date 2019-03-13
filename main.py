# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:51:34 2019

@author: Alexandre
"""

#https://nextjournal.com/gkoehler/pytorch-mnist
#https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
#http://cs231n.github.io/convolutional-networks/

import time
import os
import pandas as pd
import torch
import development
import submission


directory  = os.getcwd()

###############################################################################
#Environnement creation
###############################################################################
if not os.path.exists(directory+"/results"):
    os.makedirs(directory+"/results")
if not os.path.exists(directory+"/results/BadPredictions"):
    os.makedirs(directory+"/results/BadPredictions")
if not os.path.exists(directory+"/results/BadPredictions/training"):
    os.makedirs(directory+"/results/BadPredictions/training")
if not os.path.exists(directory+"/results/BadPredictions/validation"):
    os.makedirs(directory+"/results/BadPredictions/validation")
if not os.path.exists(directory+"/results/BadPredictions"):
    os.makedirs(directory+"/results/comparaisons")

train_images = pd.read_pickle(directory+'/input/new_train_images.pkl')
train_labels = pd.read_csv(directory+'/input/train_labels.csv')



# Parameters
compute_a_submission = False
batch_size = 100
num_epochs = 10
data_full = True
model_case = 4
loss_function = "MSE"
grad_algorithm = "SGD"
learning_rate = 1e-4
is_cuda = torch.cuda.is_available()
speed_calculs = True
save_model = False
save_bad_predictions = False
train_images = pd.read_pickle(directory+'/input/new_train_images.pkl')
train_labels = pd.read_csv(directory+'/input/train_labels.csv')


start = time.time()
accuracy = development.Neu(train_images,train_labels,batch_size,num_epochs,data_full,model_case,loss_function,grad_algorithm,learning_rate,is_cuda,speed_calculs,save_model,save_bad_predictions)
print("Took {0}s".format(round(time.time()-start,3)))
print("Final Accuracy : {0}".format(accuracy))
            
            
            
if compute_a_submission == True:
    test_images = pd.read_pickle(directory+'/input/new_test_images.pkl')
    start = time.time()
    submission.Neu(train_images,train_labels,test_images,batch_size,num_epochs,model_case,loss_function,grad_algorithm,learning_rate,is_cuda)
    print("Took {0}s".format(round(time.time()-start,3)))
