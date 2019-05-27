# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:47:05 2019

@author: Alexandre
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:51:34 2019

@author: Alexandre
"""


import sys
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import csv
import torchvision.models as models

class Flatten(torch.nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

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


def printer(ide):
    plt.figure()
    plt.title('Label: {}'.format(train_labels.iloc[ide]['Category']))
    plt.imshow(train_images[ide])
    
    
    
def dele(img):
    (x,y) = img.shape
    sm_img = np.zeros((x,y),dtype=np.uint8) 
    for i in range(x):
        for j in range(y):
            if img[i,j] == 255.0:
                sm_img[i,j] = 1 
            else:
                sm_img[i,j] = 0
    return sm_img
  

def smoother(ide):
    plt.figure()
    plt.title('Label: {}'.format(train_labels.iloc[ide]['Category']))
    plt.imshow(train_images[ide])
    img = np.copy(train_images[ide])
    img = dele(img)
    plt.figure()
    plt.title('Label: {}'.format(train_labels.iloc[ide]['Category']))
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
        try:
            if a.index(max(a)) == train_labels.iloc[ide]['Category']:
                acc += 1.
        except:
            b = [train_labels[ide,i].item() for i in range(10)]
            if a.index(max(a)) == b.index(max(b)):
                acc += 1.
    acc = acc/N
    return acc

def Accuracy_debuguage(y_pred,train_labels,images,data_set):
    N = len(train_labels)
    acc = 0.
    for ide in range(N):
        a = [y_pred[ide,i].item() for i in range(10)]
        try:
            if a.index(max(a)) == train_labels.iloc[ide]['Category']:
                acc += 1.
            else:
                plt.figure()
                plt.title('Label: {0} -> {1}'.format(train_labels.iloc[ide]['Category'],a.index(max(a))))
                plt.imshow(images[ide])
                plt.savefig(directory+"/results/BadPredictions/{0}/img_{1}.png".format(data_set,ide))
                plt.close()
        except:
            b = [train_labels[ide,i].item() for i in range(10)]
            if a.index(max(a)) == b.index(max(b)):
                acc += 1.
            else:
                plt.figure()
                plt.title('Label: {0} -> {1}'.format(b.index(max(b)),a.index(max(a))))
                plt.imshow(images[ide])
                plt.savefig(directory+"/results/BadPredictions/{0}/img_{1}.png".format(data_set,ide))
                plt.close()
    acc = acc/N
    return acc
    
    
def Neu(train_images,
        train_labels,
        batch_size = 100,
        num_epochs = 100,
        data_full = True,
        model_case = 47,
        make_a_submission = False,
        loss_function = "MSE",#"MSE", #"MSE" "Cross_Entropy"
        grad_algorithm = "adaptativeSGD", #"adaptativeSGD" "SGD" "Adadelta"
        is_cuda = torch.cuda.is_available(),
        speed_calculs = False,
        save_model = False,
        save_bad_predictions = False
        ):
    print("Model case : {0} -- Loss Function : {1} -- Gradient Algorithm : {2}".format(model_case,loss_function,grad_algorithm))
    
    x_size = train_images.shape[1]
    y_size = train_images.shape[2]
    
    if make_a_submission == False:
        if data_full == True:
    #        validation_images = train_images[35000:40000]
    #        validation_labels = train_labels[35000:40000]
            validation_images = [train_images[35000+batch_size*i:35000+batch_size*(i+1)] for i in range(int(5000/batch_size))]
            validation_labels = [train_labels[35000+batch_size*i:35000+batch_size*(i+1)] for i in range(int(5000/batch_size))]
            train_images = [train_images[batch_size*i:batch_size*(i+1)] for i in range(int(35000/batch_size))]
            train_labels = [train_labels[batch_size*i:batch_size*(i+1)] for i in range(int(35000/batch_size))]
        else:
            batch_size = 20
    #        validation_images = train_images[39900:40000]
    #        validation_labels = train_labels[39900:40000]
            validation_images = [train_images[39900+batch_size*i:39900+batch_size*(i+1)] for i in range(int(100/batch_size))]
            validation_labels = [train_labels[39900+batch_size*i:39900+batch_size*(i+1)] for i in range(int(100/batch_size))]
            train_images = [train_images[batch_size*i:batch_size*(i+1)] for i in range(int(1000/batch_size))]
            train_labels = [train_labels[batch_size*i:batch_size*(i+1)] for i in range(int(1000/batch_size))]

        if is_cuda==True:
            train_images = [torch.cuda.FloatTensor(images.reshape(batch_size, 1, x_size, y_size)) for images in train_images]
            train_labels = [torch.cuda.FloatTensor([vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in train_labels]
        else:
            train_images = [torch.tensor(images.reshape(batch_size, 1, x_size, y_size)) for images in train_images]
            train_labels = [torch.tensor([vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in train_labels]

        if is_cuda==True:
            x_val = [torch.cuda.FloatTensor(images.reshape(batch_size, 1, x_size, y_size)) for images in validation_images]
    #        validation_labels = [torch.cuda.FloatTensor([vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in validation_labels]
        else:
            x_val = [torch.tensor(images.reshape(batch_size, 1, x_size, y_size)) for images in validation_images]
    #        validation_labels = [torch.tensor([vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in validation_labels]
       
    
    else:
        num_epochs += 5
        test_images = pd.read_pickle(directory+'/input/new_test_images.pkl')
        test_images = [test_images[batch_size*i:batch_size*(i+1)] for i in range(int(10000/batch_size))]
        train_images = [train_images[batch_size*i:batch_size*(i+1)] for i in range(int(40000/batch_size))]
        train_labels = [train_labels[batch_size*i:batch_size*(i+1)] for i in range(int(40000/batch_size))]
        if is_cuda==True:
            train_images = [torch.cuda.FloatTensor(images.reshape(batch_size, 1, x_size, y_size)) for images in train_images]
            train_labels = [torch.cuda.FloatTensor([vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in train_labels]
        else:
            train_images = [torch.tensor(images.reshape(batch_size, 1, x_size, y_size)) for images in train_images]
            train_labels = [torch.tensor([vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in train_labels]

        if is_cuda==True:
            test_images = [torch.cuda.FloatTensor(images.reshape(batch_size, 1, x_size, y_size)) for images in test_images]
        else:
            test_images = [torch.tensor(images.reshape(batch_size, 1, x_size, y_size)) for images in test_images]

        if is_cuda==True:
            test_images = [torch.cuda.FloatTensor(images.reshape(batch_size, 1, x_size, y_size)) for images in test_images]
        else:
            test_images = [torch.tensor(images.reshape(batch_size, 1, x_size, y_size)) for images in test_images]
              
    num_data = len(train_images)
    data_training = [(train_images[i],train_labels[i]) for i in range(num_data)]

    
    
    if model_case == 1:
        model = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(51*51, 1048),
            torch.nn.ReLU(),
            torch.nn.Linear(1048, 10),
        )
        
    elif model_case == 2:
        model = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            Flatten(),
            torch.nn.Linear(51*51, 1048),
            torch.nn.ReLU(),
            torch.nn.Linear(1048, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )
    
    elif model_case == 3:
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 32, 3),      #[batch_size, 32, 48, 48]
                                torch.nn.ReLU(),                #[batch_size, 32, 48, 48]
                                torch.nn.Conv2d(32, 64, 3),     #[batch_size, 64, 45, 45]
                                torch.nn.ReLU(),                #[batch_size, 64, 45, 45]
                                torch.nn.MaxPool2d(2),          #[batch_size, 64, 23, 23]
                                torch.nn.Dropout(p=0.25),       #[batch_size, 64, 23, 23]
                                Flatten(),                      #[batch_size, 64*23*23]
                                torch.nn.Linear(64*23*23,128),  #[batch_size, 128]
                                torch.nn.ReLU(),                #[batch_size, 128]
                                torch.nn.Dropout(p=0.5),        #[batch_size, 128]
                                torch.nn.Linear(128,10),        #[batch_size, 10]
                                torch.nn.Softmax()              #[batch_size, 10]
                                )
    
    elif model_case == 4:
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 32, 5),       #[batch_size, 6, 46, 46]
                                torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(32, 64, 5),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(64*9*9,120),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(120,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                )
        
    elif model_case == 41: #0.9479 after 13 epochs
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 128, 5),       #[batch_size, 6, 46, 46]
                                torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(128, 256, 5),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(256*9*9,256),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(256,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                )
        
    elif model_case == 42: #0.9510 after 32 epochs -- Kaggle 0.9533
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 128, 5),       #[batch_size, 6, 46, 46]
                                torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(128, 512, 5),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(512*9*9,120),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(120,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                )
        
    elif model_case == 43: #0.9507 after 37 epochs
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 128, 5),       #[batch_size, 6, 46, 46]
           #                     torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(128, 512, 5),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(512*9*9,120),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(120,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                )
        
    elif model_case == 44: #0.9508 after 13 epochs
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 256, 5),       #[batch_size, 6, 46, 46]
                                torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(256, 512, 5),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(512*9*9,120),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(120,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                )
        
    elif model_case == 45: #0.9508 after 13 epochs
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 6, 4),       #[batch_size, 6, 46, 46]
                                torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(6, 16, 4),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(16*10*10,120),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(120,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                )
    
    elif model_case == 46: #0.9508 after 18 epochs
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 256, 5),       #[batch_size, 6, 46, 46]
                                torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(256, 1024, 5),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(1024*9*9,256),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(256,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                )
        
    elif model_case == 47: #0.9536 after 33 epochs & #0.955 after 14 epochs
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 128, 5),       #[batch_size, 6, 46, 46]
                                torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(128, 1024, 5),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(1024*9*9,256),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(256,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                
                                )
        
    # Model case : 475 -- Loss Function : Cross_Entropy -- Gradient Algorithm : Adadelta    
    elif model_case == 475: #0.948 after 38 epochs
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 128, 5),       #[batch_size, 6, 46, 46]
                                torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(128, 1024, 5),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(1024*9*9,256),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(256,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                torch.nn.Softmax(),
                                )
                          
    elif model_case == 48: #0.9540 after 15 epochs BUT ~7000s
        model = torch.nn.Sequential(                            #[batch_size, 1, 51, 51]
                                torch.nn.Conv2d(1, 128, 5),       #[batch_size, 6, 46, 46]
                                torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(128, 2048, 5),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(2048*9*9,512),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(512,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                )
                
    elif model_case == 55:
        model = torch.nn.Sequential(                            
                                torch.nn.Conv2d(1, 64, kernel_size=7, stride=3),
                                torch.nn.ReLU(),                
                                torch.nn.MaxPool2d(3,stride=2),       
                                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),     
                                torch.nn.ReLU(),                
                                torch.nn.Conv2d(128, 32, kernel_size=3, stride=1),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(3, stride=2),       
                                Flatten(),                      
                                torch.nn.Linear(32,128),       
                                torch.nn.ReLU(),                
                                torch.nn.Linear(128,10),         
                                )
        
    elif model_case == 5:
        model = torch.nn.Sequential(                            #[batch_size, 1, 64, 64]
                                torch.nn.Conv2d(1, 32, 2),      #[batch_size, 32, 60, 60]
                                torch.nn.ReLU(),                #[batch_size, 32, 60, 60]
                                torch.nn.MaxPool2d(2, 2),
                                torch.nn.Conv2d(32, 64, 2),     #[batch_size, 64, 26, 26]
                                torch.nn.ReLU(),                #[batch_size, 64, 26, 26]
                                torch.nn.MaxPool2d(2, 2),
                                torch.nn.Conv2d(64, 128, 2),    #[batch_size, 128, 9, 9]
                                torch.nn.ReLU(),                #[batch_size, 128, 9, 9]
                                torch.nn.MaxPool2d(2, 2),
                                torch.nn.Conv2d(128, 256, 2),    #[batch_size, 128, 9, 9]
                                torch.nn.ReLU(),                #[batch_size, 128, 9, 9]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 128, 5, 5]
                                torch.nn.Conv2d(256, 512, 2),    #[batch_size, 128, 9, 9]
                                torch.nn.ReLU(),                #[batch_size, 128, 9, 9]
                                Flatten(),                      #[batch_size, 16*5*5]
          #                      torch.nn.Linear(512*2*2,512),   #[batch_size, 120]
          #                      torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(512,64),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(64,10),         #[batch_size, 10]
                                torch.nn.Softmax(),
                                )

    if is_cuda==True:
        model = model.cuda()
    
    if loss_function == "MSE":
        loss_fn = torch.nn.MSELoss(reduction='sum')
    elif loss_function == "Cross_Entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
        
    losses = []
    timer = []
    if speed_calculs == False:
        accuracies = []
        accuracies_val = []
        acc_max = 0
        epoch_max = 0
        timer_val = []
    
    if grad_algorithm == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    elif grad_algorithm == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    elif grad_algorithm == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters())
        
    
    for epoch in range(num_epochs):
      
        if grad_algorithm == "adaptativeSGD":
            if (epoch < 2) :
                optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
            elif (epoch < 2+3):
                optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)
            elif (epoch < 2+3+3):
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
            elif (epoch < 2+3+3+4):
                optimizer = torch.optim.SGD(model.parameters(), lr=5e-5, momentum=0.9)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
        
        for t, (x,y) in enumerate(data_training):
            
            if speed_calculs == False:
                if t == 0:
                    if epoch != 0:
                        accv = [Accuracy(model(x_val[t]),validation_labels[t]) for t in range(len(x_val))]
                        accv = sum(accv)/len(accv)
                        accuracies_val.append(accv)
                        timer_val.append((t+epoch*num_data)/(num_data-1))
                        if accv > acc_max:
                            acc_max = accv
                            epoch_max = epoch
        
            y_pred = model(x)
    
            if loss_function == "MSE":
                loss = loss_fn(y_pred, y)
            elif loss_function == "Cross_Entropy":
                loss = loss_fn(y_pred, torch.max(y.long(), 1)[1])
                
            losses.append(loss.data.item())
            timer.append((t+epoch*num_data)/(num_data-1))
            
            if speed_calculs == False:
                if t == 0:
                    if epoch != 0:
                        acc = [Accuracy(model(train_images[t]),train_labels[t]) for t in range(num_data)]
                        acc = sum(acc)/len(acc)
                        accuracies.append(acc)
            
            if speed_calculs == False:
                if epoch != 0:
                    ph = "\rEpoch [{0}/{1}], Step [{2}/{3}] -- Loss: {4} -- Accuracy : t->{5} & v->{6}    ".format(epoch,num_epochs-1,t,num_data-1,round(loss.item(),4),round(acc,4),round(accv,4))
                else:
                    ph = "\rEpoch [{0}/{1}], Step [{2}/{3}] -- Loss: {4} -- Accuracy : t->0.1000 & v->0.1000    ".format(epoch,num_epochs-1,t,num_data-1,round(loss.item(),4))
                sys.stdout.write(ph)
                sys.stdout.flush()
            else:
                ph = "\rEpoch [{0}/{1}], Step [{2}/{3}] -- Loss : {4}    ".format(epoch,num_epochs-1,t,num_data-1,round(loss.item(),4))
                sys.stdout.write(ph)
                sys.stdout.flush()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
                
    if speed_calculs == False:
        fig, ax1 = plt.subplots(figsize=None)
        ax2 = ax1.twinx()
        plt.title('Loss & Accuracy over time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss',color="blue")
        ax1.plot(timer,losses,"blue",label="Loss")
        for tl in ax1.get_yticklabels():
            tl.set_color("blue")
        ax2.set_ylabel('Accuracy')
        ax2.plot(timer_val,accuracies,"red",label="training")
        ax2.plot(timer_val,accuracies_val,"green",label="validation")
        ax2.legend()
        plt.savefig(directory+"/results/fig.png")
        plt.show()
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=None)
        plt.title('Loss over time')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss',color="blue")
        ax.plot(losses,"blue",label="Loss")
        for tl in ax.get_yticklabels():
            tl.set_color("blue")
        plt.savefig(directory+"/results/fig.png")
        plt.show()
        plt.close()
    
    ###########################################################################
    #Save the model
    ###########################################################################
    if save_model == True:
        torch.save(model.state_dict(), directory+"/results/model.pth")
        
    if save_bad_predictions == True:
        y_pred = model(x)
        Accuracy_debuguage(y_pred,train_labels,train_images,"training")
        y_val_pred = model(x_val)
        Accuracy_debuguage(y_val_pred,validation_labels,validation_images,"validation")
    
    if make_a_submission == True:
        with open(directory+'/results/submission.csv', 'w') as file:
            file_writer = csv.writer(file, delimiter=',')
            file_writer.writerow(['Id', 'Category'])

            for t in range(len(test_images)):
                y = model(test_images[t])
                for ide in range(batch_size):
                    a = [y[ide,i].item() for i in range(10)]
                    a.index(max(a))
                    file_writer.writerow([ide+t*100, a.index(max(a))])
        print("----- done -----")
        return (1.,1.)
                    
                    
    else:
#        acc = [Accuracy(model(x_val[t]),validation_labels[t]) for t in range(len(x_val))]
#        (sum(acc)/len(acc))
        
        return (acc_max,epoch_max)
        

start = time.time()
(accuracy,epoch) = Neu(train_images,train_labels)
print("Took {0}s".format(time.time()-start))
print("Best Accuracy : {0} after {1} epochs".format(accuracy,epoch))

