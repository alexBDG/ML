# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:51:34 2019

@author: Alexandre
"""

#https://nextjournal.com/gkoehler/pytorch-mnist
#https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a

import sys
import time
import os

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


directory  = os.getcwd()

#Files are stored in pickle format.
#Load them like how you load any pickle. The data is a numpy array
import pandas as pd
train_images = pd.read_pickle(directory+'/input/new_train_images.pkl')
train_labels = pd.read_csv(directory+'/input/train_labels.csv')

train_images.shape




import matplotlib.pyplot as plt
def printer(ide):
    fig = plt.figure()
    plt.title('Label: {}'.format(train_labels.iloc[ide]['Category']))
    plt.imshow(train_images[ide])
    
    
def smoothing(img):
    sm_img = np.copy(img)
    for i in range(1,63):
        sm_img[i,0] = (img[i-1,0]+img[i+1,0]-3*img[i,0]+img[i,1])/6
        for j in range(1,63):
            sm_img[i,j] = (img[i-1,j]+img[i+1,j]-4*img[i,j]+img[i,j+1]+img[i,j-1])/8
        sm_img[i,63] = (img[i-1,63]+img[i+1,63]-3*img[i,63]+img[i,62])/6
    for j in range(63):
        sm_img[0,j] = (img[1,j]-3*img[0,j]+img[0,j+1]+img[0,j-1])/6
        sm_img[63,j] = (img[62,j]-3*img[63,j]+img[63,j+1]+img[63,j-1])/6
    return sm_img

def smooth(maxi):
    fig = plt.figure()
    plt.title('Label: {}'.format(train_labels.iloc[18]['Category']))
    plt.imshow(train_images[18])
    img = np.copy(train_images[18])
    for k in range(maxi):
        img += 10**(-2)*smoothing(img)
    fig = plt.figure()
    plt.title('Label: {}'.format(train_labels.iloc[18]['Category']))
    plt.imshow(img)
    
#smooth(150)
    np.zeros()
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
    fig = plt.figure()
    plt.title('Label: {}'.format(train_labels.iloc[ide]['Category']))
    plt.imshow(train_images[ide])
    img = np.copy(train_images[ide])
    img = dele(img)
    fig = plt.figure()
    plt.title('Label: {}'.format(train_labels.iloc[ide]['Category']))
    plt.imshow(img)
    
def biggest(img):
    
    return 0.
    
#smoother(654)


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

def Accuracy_debuguage(y_pred,train_labels,train_images,data_set):
    N = len(train_labels)
    acc = 0.
    for ide in range(N):
        a = [y_pred[ide,i].item() for i in range(10)]
        if a.index(max(a)) == train_labels.iloc[ide]['Category']:
            acc += 1.
        else:
            plt.figure()
            plt.title('Label: {0} -> {1}'.format(train_labels.iloc[ide]['Category'],a.index(max(a))))
            plt.imshow(train_images[ide])
            plt.savefig(directory+"/results/BadPredictions/{0}/img_{1}.png".format(data_set,ide))
#            plt.imsave(directory+"/results/BadPredictions/{0}/img_{1}.png".format(data_set,ide),train_images[ide])
            plt.close()
    acc = acc/N
    return acc
    
    
import torch

def Neu(train_images,train_labels):
    
    validation_images = train_images[35000:]
    validation_labels = train_labels[35000:]

    train_images = train_images[:5000]
    train_labels = train_labels[:5000]
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    
    N, D_in, H_1, H_2, D_out = len(train_images), train_images.shape[2]*train_images.shape[1], 512, 32, 10
    
    # Create random Tensors to hold inputs and outputs
    #x = torch.randn(N, D_in)
    #y = torch.randn(N, D_out)
#    for ide in range(N):
#        ph = "\rProgression: {0} % ".format(round(float(100*ide)/float(N-1),3))
#        sys.stdout.write(ph)
#        sys.stdout.flush()
#        train_images[ide] = dele(train_images[ide])
#    print()
        
    x = torch.tensor([train_images[ide].reshape(D_in) for ide in range(N)])
    y = torch.tensor([vect(train_labels.iloc[ide]['Category']) for ide in range(N)])
    
    Nval = len(validation_images)
#    for ide in range(Nval):
#        ph = "\rProgression: {0} % ".format(round(float(100*ide)/float(Nval-1),3))
#        sys.stdout.write(ph)
#        sys.stdout.flush()
#        validation_images[ide] = dele(validation_images[ide])
#    print()

    
    x_val = torch.tensor([validation_images[ide].reshape(D_in) for ide in range(Nval)])
    y_val = torch.tensor([vect(validation_labels.iloc[ide]['Category']) for ide in range(Nval)])

    
    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H_1),
        torch.nn.ReLU(),
#        torch.nn.Linear(H_1, H_2),
#        torch.nn.ReLU(),
#        torch.nn.Sigmoid(),
#        torch.nn.Softplus(),
        torch.nn.Linear(H_1, D_out),
    )
    
    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(reduction='sum')
#    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    accuracies_val = []
    iterr = 750
    lr = 1e-5
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
#    optimizer = torch.optim.Adam(model.parameters(), lr)
    try:
        for t in range(iterr):
            # Forward pass: compute predicted y by passing x to the model. Module objects
            # override the __call__ operator so you can call them like functions. When
            # doing so you pass a Tensor of input data to the Module and it produces
            # a Tensor of output data.
            y_val_pred = model(x_val)
            accv = Accuracy(y_val_pred,validation_labels)
            accuracies_val.append(accv)
        
            y_pred = model(x)
    
            # Compute and print loss. We pass Tensors containing the predicted and true
            # values of y, and the loss function returns a Tensor containing the
            # loss.
            loss = loss_fn(y_pred, y)
    #        loss = loss_fn(y_pred, torch.max(y.long(), 1)[1])
            losses.append(loss.data.item())
            acc = Accuracy(y_pred,train_labels)
            accuracies.append(acc)
            
            ph = "\rProgression: {0} % -- Loss : {1} -- Accuracy : t->{2} & v->{3}    ".format(round(float(100*t)/float(iterr-1),3),round(loss.item(),2),round(acc,2),round(accv,2))
            sys.stdout.write(ph)
            sys.stdout.flush()
    
            # Zero the gradients before running the backward pass.
    #            model.zero_grad()
            optimizer.zero_grad()
    
            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            loss.backward()
    
            # Update the weights using gradient descent. Each parameter is a Tensor, so
            # we can access its gradients like we did before.
    #            with torch.no_grad():
    #                for param in model.parameters():
    #                    param -= learning_rate * param.grad
                    
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
    except:
        print("STOP")
    print()
                
    import matplotlib.pyplot as plt
    #fig = plt.figure()
    #plt.title('Loss over time')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.plot(losses)
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
    
    ###########################################################################
    #Save the model
    ###########################################################################
    torch.save(model.state_dict(), directory+"/results/model.pth")
    Accuracy_debuguage(y_pred,train_labels,train_images,"training")
    Accuracy_debuguage(y_pred,validation_labels,validation_images,"validation")
        
    return (y_val,y_val_pred)
        
start = time.time()
(y_val,y_val_pred) = Neu(train_images,train_labels)
print("Took {0}s".format(time.time()-start))

def make_a_pred(x,y):
    fig = plt.figure()
    a = [y_val_pred[ide,i].item() for i in range(10)]
    b = [y_val[ide,i].item() for i in range(10)]
    plt.title('Label: {0} -> {1}'.format(b.index(max(b)),a.index(max(a))))
    plt.imshow(train_images[ide])

#make_a_pred(2) #doesn't works
