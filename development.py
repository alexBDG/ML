# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:10:39 2019

@author: Alexandre
"""

import sys
import matplotlib.pyplot as plt
import torch
import functions
import modules
import os


directory  = os.getcwd()

def Neu(train_images,
        train_labels,
        batch_size = 100,
        num_epochs = 10,
        data_full = False,
        model_case = 3,
        loss_function = "MSE",
        grad_algorithm = "SGD",
        learning_rate = 1e-4,
        is_cuda = torch.cuda.is_available(),
        speed_calculs = True,
        save_model = False,
        save_bad_predictions = False
        ):
    
    x_size = train_images.shape[1]
    y_size = train_images.shape[2]
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
        train_labels = [torch.cuda.FloatTensor([functions.vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in train_labels]
    else:
        train_images = [torch.tensor(images.reshape(batch_size, 1, x_size, y_size)) for images in train_images]
        train_labels = [torch.tensor([functions.vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in train_labels]

    if is_cuda==True:
        x_val = [torch.cuda.FloatTensor(images.reshape(batch_size, 1, x_size, y_size)) for images in validation_images]
#        validation_labels = [torch.cuda.FloatTensor([vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in validation_labels]
    else:
        x_val = [torch.tensor(images.reshape(batch_size, 1, x_size, y_size)) for images in validation_images]
#        validation_labels = [torch.tensor([vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in validation_labels]
        
    num_data = len(train_images)
    data_training = [(train_images[i],train_labels[i]) for i in range(num_data)]
    
    
    if model_case == 1:
        model = modules.Net_1()
    elif model_case == 2:
        model = modules.Net_2()
    elif model_case == 3:
        model = modules.Net_3()
    elif model_case == 4:
        model = modules.Net_4()


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
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)
    elif grad_algorithm == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    elif grad_algorithm == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters())
        
        
    for epoch in range(num_epochs):
        
        for t, (x,y) in enumerate(data_training):
            
            if speed_calculs == False:
                y_val_pred = model(x_val[t])
                accv = functions.Accuracy(y_val_pred,validation_labels[t])
                accuracies_val.append(accv)
        
            y_pred = model(x)
    
            if loss_function == "MSE":
                loss = loss_fn(y_pred, y)
            elif loss_function == "Cross_Entropy":
                loss = loss_fn(y_pred, torch.max(y.long(), 1)[1])
                
            losses.append(loss.data.item())
            
            if speed_calculs == False:
                acc = functions.Accuracy(y_pred,train_labels[t])
                accuracies.append(acc)
            
            if speed_calculs == False:
                ph = "\rEpoch [{0}/{1}], Step [{2}/{3}] -- Loss: {4} -- Accuracy : t->{2} & v->{3}    ".format(epoch,num_epochs-1,t,num_data-1,round(loss.item(),4),round(acc,2),round(accv,2))
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
        functions.Accuracy_debuguage(y_pred,train_labels,train_images,"training")
        y_val_pred = model(x_val)
        functions.Accuracy_debuguage(y_val_pred,validation_labels,validation_images,"validation")
        
    acc = [functions.Accuracy(model(x_val[t]),validation_labels[t]) for t in range(len(x_val))]
        
    return (sum(acc)/len(acc))
