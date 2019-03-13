# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:00:32 2019

@author: Alexandre
"""

import sys
import matplotlib.pyplot as plt
import torch
import csv
import functions
import modules
import os


directory  = os.getcwd()

def Neu(train_images,
        train_labels,
        test_images,
        batch_size = 100,
        num_epochs = 50,
        model_case = 4,
        loss_function = "MSE",
        grad_algorithm = "SGD",
        learning_rate = 1e-4,
        is_cuda = torch.cuda.is_available(),
        ):
    
    x_size = train_images.shape[1]
    y_size = train_images.shape[2]
    test_images = [test_images[batch_size*i:batch_size*(i+1)] for i in range(int(10000/batch_size))]        
    train_images = [train_images[batch_size*i:batch_size*(i+1)] for i in range(int(40000/batch_size))]
    train_labels = [train_labels[batch_size*i:batch_size*(i+1)] for i in range(int(40000/batch_size))]
        
    if is_cuda==True:
        train_images = [torch.cuda.FloatTensor(images.reshape(batch_size, 1, x_size, y_size)) for images in train_images]
        train_labels = [torch.cuda.FloatTensor([functions.vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in train_labels]
    else:
        train_images = [torch.tensor(images.reshape(batch_size, 1, x_size, y_size)) for images in train_images]
        train_labels = [torch.tensor([functions.vect(labels.iloc[ide]['Category']) for ide in range(batch_size)]) for labels in train_labels]

    if is_cuda==True:
        test_images = [torch.cuda.FloatTensor(images.reshape(batch_size, 1, x_size, y_size)) for images in test_images]
    else:
        test_images = [torch.tensor(images.reshape(batch_size, 1, x_size, y_size)) for images in test_images]
        
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
    
    if grad_algorithm == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)
    elif grad_algorithm == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    elif grad_algorithm == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters())
        
        
    for epoch in range(num_epochs):
        
        for t, (x,y) in enumerate(data_training):
                    
            y_pred = model(x)
    
            if loss_function == "MSE":
                loss = loss_fn(y_pred, y)
            elif loss_function == "Cross_Entropy":
                loss = loss_fn(y_pred, torch.max(y.long(), 1)[1])
                
            losses.append(loss.data.item())
                        
            ph = "\rEpoch [{0}/{1}], Step [{2}/{3}] -- Loss : {4}    ".format(epoch,num_epochs-1,t,num_data-1,round(loss.item(),4))
            sys.stdout.write(ph)
            sys.stdout.flush()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
                
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
                    
    with open('submission.csv', 'w') as file:
        file_writer = csv.writer(file, delimiter=',')
        file_writer.writerow(['Id', 'Category'])
        
        for t in range(len(test_images)):
            y = model(test_images[t])
            for ide in range(batch_size):
                a = [y[ide,i].item() for i in range(10)]
                a.index(max(a))
                file_writer.writerow([ide+t*10, a.index(max(a))])
        
    print("----- done -----")
