# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:51:34 2019

@author: Alexandre
"""

#https://nextjournal.com/gkoehler/pytorch-mnist
#https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
#http://cs231n.github.io/convolutional-networks/

import sys
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
    
    
def Neu(train_images,
        train_labels,
        data_full = True,
        model_case = 4,
        loss_function = "MSE",
        grad_algorithm = "SGD",
        iterration = 2000,
        is_cuda = torch.cuda.is_available(),
        speed_calculs = True,
        save_model = True,
        save_bad_predictions = False
        ):
    
    if data_full == True:
        validation_images = train_images[35000:40000]
        validation_labels = train_labels[35000:40000]
        train_images = train_images[:35000]
        train_labels = train_labels[:35000]
        
    else:
        validation_images = train_images[39900:40000]
        validation_labels = train_labels[39900:40000]
        train_images = train_images[:1000]
        train_labels = train_labels[:1000]
        
        
    batch_size = len(train_images)
    if is_cuda==True:
        x = torch.cuda.FloatTensor(train_images.reshape(batch_size, 1, train_images.shape[1], train_images.shape[2]))
        y = torch.cuda.FloatTensor([vect(train_labels.iloc[ide]['Category']) for ide in range(batch_size)])
    else:
        x = torch.tensor(train_images.reshape(batch_size, 1, train_images.shape[1], train_images.shape[2]))
        y = torch.tensor([vect(train_labels.iloc[ide]['Category']) for ide in range(batch_size)])

    
    batch_size_val = len(validation_images)
    if is_cuda==True:
        x_val = torch.cuda.FloatTensor(validation_images.reshape(batch_size_val, 1, validation_images.shape[1], validation_images.shape[2]))
    else:
        x_val = torch.tensor(validation_images.reshape(batch_size_val, 1, validation_images.shape[1], validation_images.shape[2]))

    
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
                                torch.nn.Conv2d(1, 6, 5),       #[batch_size, 6, 46, 46]
                                torch.nn.ReLU(),                #[batch_size, 6, 46, 46]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 6, 23, 23]
                                torch.nn.Conv2d(6, 16, 5),      #[batch_size, 16, 18, 18]
                                torch.nn.ReLU(),                #[batch_size, 16, 18, 18]
                                torch.nn.MaxPool2d(2, 2),       #[batch_size, 16, 9, 9]
                                Flatten(),                      #[batch_size, 16*9*9]
                                torch.nn.Linear(16*9*9,120),    #[batch_size, 120]
                                torch.nn.ReLU(),                #[batch_size, 120]
                                torch.nn.Linear(120,84),        #[batch_size, 84]
                                torch.nn.ReLU(),                #[batch_size, 84]
                                torch.nn.Linear(84,10),         #[batch_size, 10]
                                )

    if is_cuda==True:
        model = model.cuda()
    
    if loss_function == "MSE":
        loss_fn = torch.nn.MSELoss(reduction='sum')
    elif loss_function == "Cross_Entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
        
    losses = []
    if speed_calculs == False:
        accuracies = []
        accuracies_val = []
    
    if grad_algorithm == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    elif grad_algorithm == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    elif grad_algorithm == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters())
        
        
    try:
        for t in range(iterration):
            
            if speed_calculs == False:
                y_val_pred = model(x_val)
                accv = Accuracy(y_val_pred,validation_labels)
                accuracies_val.append(accv)
        
            y_pred = model(x)
    
            if loss_function == "MSE":
                loss = loss_fn(y_pred, y)
            elif loss_function == "Cross_Entropy":
                loss = loss_fn(y_pred, torch.max(y.long(), 1)[1])
                
            losses.append(loss.data.item())
            
            if speed_calculs == False:
                acc = Accuracy(y_pred,train_labels)
                accuracies.append(acc)
            
            if speed_calculs == False:
                ph = "\rProgression: {0} % -- Loss : {1} -- Accuracy : t->{2} & v->{3}    ".format(round(float(100*t)/float(iterration-1),3),round(loss.item(),2),round(acc,2),round(accv,2))
                sys.stdout.write(ph)
                sys.stdout.flush()
            else:
                ph = "\rProgression: {0} % -- Loss : {1}".format(round(float(100*t)/float(iterration-1),3),round(loss.item(),2))
                sys.stdout.write(ph)
                sys.stdout.flush()
    
            optimizer.zero_grad()
    
            loss.backward()
    
            optimizer.step()
            
    except:
        print()
        print("STOP")
                
    if speed_calculs == False:
        fig, ax1 = plt.subplots(figsize=None)
        ax2 = ax1.twinx()
        plt.title('Loss & Accuracy over time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss',color="blue")
        ax1.plot(losses,"blue",label="Loss")
        for tl in ax1.get_yticklabels():
            tl.set_color("blue")
        ax2.set_ylabel('Accuracy')
        ax2.plot(accuracies,"red",label="training")
        ax2.plot(accuracies_val,"green",label="validation")
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
        
    return Accuracy(y_pred,train_labels)
        
start = time.time()
accuracy = Neu(train_images,train_labels)
print("Took {0}s".format(time.time()-start))
print("Final Accuracy : {0}".format(accuracy))

    
def model_loader():
    model = torch.nn.Sequential(
                                    torch.nn.Linear(51*51, 10),
                                    torch.nn.ReLU(),
                                )
    model.load_state_dict(torch.load(directory+"/results/model.pth"))
    model.eval()
