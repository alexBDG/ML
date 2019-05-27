import cv2
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import time
import pickle
import os
from scipy import ndimage



directory  = os.getcwd()
#train_images = pd.read_pickle(directory+'/input/train_images.pkl')
test_images = pd.read_pickle(directory+'/input/test_images.pkl')
train_labels = pd.read_csv(directory+'/input/train_labels.csv')

def printer(ide):
    plt.figure()
    plt.title('Label: {}'.format(train_labels.iloc[ide]['Category']))
    plt.imshow(train_images[ide])
    plt.close
    
def printing(img):
    plt.figure()
    plt.imshow(img)
    plt.close
    



def dele(img):
    sm_img = np.zeros((64,64),dtype=np.uint8) 
    for i in range(64):
        for j in range(64):
            if img[i,j] == 255.0:
                sm_img[i,j] = 0 
            else:
                sm_img[i,j] = 255
    return sm_img

def dele_change(img):
    (x,y) = img.shape
    sm_img = np.zeros((x,y),dtype=np.uint8) 
    for i in range(x):
        for j in range(y):
            if img[i,j] != 255:
                sm_img[i,j] = 1
    return sm_img

def dele_norm(ide):
    img = train_images[ide]
    (x,y) = img.shape
    sm_img = np.zeros((x,y),dtype=np.uint8) 
    for i in range(x):
        for j in range(y):
            if img[i,j] == 255.0:
                sm_img[i,j] = 1 
            else:
                sm_img[i,j] = 0
    return sm_img



def decomposition(ide):

    img = np.zeros((64,64,3), dtype = np.uint8)
    img_cleaned = dele(test_images[ide])
    for k in range(3):
        img[:,:,k] = img_cleaned
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    # threshold to get just the signature (INVERTED)
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
    
    image, contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    maxi = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
#        ((x2,y2),(w2,h2),r2) = cv2.minAreaRect(cont)
        ################################
        #TO find the max area
        ################################
    #    area = w*h
    #    if area > mx_area:
    #        mx = x,y,w,h
    #        mx_area = area
        ################################
        #TO find the max lenght
        ################################
#        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        lenght = max(w,h)
        if lenght > maxi:
            mx = x,y,w,h
            maxi = lenght
#            cont_max = cont
#    return maxi
            
#    rect = cv2.minAreaRect(cont_max)
#    print(rect)
#    box = cv2.boxPoints(rect)
#    box = np.int0(box)
#    cv2.drawContours(img,[box],0,(0,0,255),2)
#    return img
        
    x,y,w,h = mx
    
    # Output to files
#    printer(ide)
    roi=img[y:y+h,x:x+w]
    
    roi_final = dele_change(roi[:,:,0])
#    fig = plt.figure()
#    plt.imshow(roi_final)
    
    new_img = np.zeros((32,32), dtype = np.uint8)
    (x_c,y_c) = (16-int(h/2),16-int(w/2))
    try:
        new_img[x_c:x_c+h,y_c:y_c+w] = roi_final
    except:
        ct_ms = ndimage.measurements.center_of_mass(roi_final)
#        plt.figure()
#        plt.imshow(roi_final)
#        plt.savefig(directory+'/input/train/fig_{0}.png'.format(ide))
#        plt.close()
#        time.sleep(5)
        
        if w > 32:
            if h > 32:
                if ct_ms[0] < w/2:
                    if ct_ms[1] < h/2:
                        new_img = roi_final[0:32,0:32]
                    else:
                        new_img = roi_final[h-32:h,0:32]
                else:
                    if ct_ms[1] < h/2:
                        new_img = roi_final[0:32,w-32:w]
                    else:
                        new_img = roi_final[h-32:h,w-32:w]
            else:    
                if ct_ms[0] < w/2:        
                    new_img[x_c:x_c+h,0:32] = roi_final[:,0:32]
                else:
                    new_img[x_c:x_c+h,0:32] = roi_final[:,w-32:w]
        else:
            if ct_ms[1] < h/2:        
                new_img[0:32,y_c:y_c+w] = roi_final[0:32,:]
            else:        
                new_img[0:32,y_c:y_c+w] = roi_final[h-32:h,:]
#        plt.figure()
#        plt.imshow(new_img)
#        plt.savefig(directory+'/input/train/fig_{0}_bis.png'.format(ide))
#        plt.close()
#        time.sleep(5)
                
        
        return ide
        
    
    return new_img


#ndimage.measurements.center_of_mass(a)
            


def decomposition_MINST(ide):

    img = np.zeros((64,64,3), dtype = np.uint8)
    img_cleaned = dele(train_images[ide])
    for k in range(3):
        img[:,:,k] = img_cleaned
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    # threshold to get just the signature (INVERTED)
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
    
    image, contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find object with the biggest bounding box
    mx = (0,0,0,0)
    maxi = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        ((x2,y2),(w2,h2),r2) = cv2.minAreaRect(cont)
        lenght = max(w2,h2)
        if lenght > maxi:
            mx = x,y,w,h
            maxi = lenght
            
        
    x,y,w,h = mx
    
    roi=img[y:y+h,x:x+w]
    
    roi_final = dele_change(roi[:,:,0])
    
    new_img = np.zeros((52,52), dtype = np.uint8)
    (x_n,y_n) = ndimage.measurements.center_of_mass(roi_final)
    (x_c,y_c) = (int(26-x_n),int(26-y_n))
    try:
        new_img[x_c:x_c+h,y_c:y_c+w] = roi_final
    except:
        try:
            print(new_img[x_c-1:x_c+h,y_c:y_c+w].shape)
            new_img[x_c-1:x_c+h,y_c:y_c+w] = roi_final
        except:
            try:
                new_img[x_c+1:x_c+h,y_c:y_c+w] = roi_final
            except:
                try:
                    new_img[x_c:x_c+h,y_c-1:y_c+w] = roi_final
                except:
                    try:
                        new_img[x_c:x_c+h,y_c+1:y_c+w] = roi_final
                    except:
                        try:
                            new_img[x_c-1:x_c+h,y_c-1:y_c+w] = roi_final
                        except:
                            new_img[x_c+1:x_c+h,y_c+1:y_c+w] = roi_final
    
    return new_img


def Find_max_length():
    total_maxi = 0
    ide_max = 0
    list_max = []
    for ide in range(len(train_images)):
        ph = "\rProgression: {0} % ".format(round(float(100*ide)/float(len(train_images)-1),3))
        sys.stdout.write(ph)
        sys.stdout.flush()
        maxi = decomposition(ide)
        if maxi > 28:#total_maxi:
            total_maxi = maxi
            ide_max = ide
            list_max += [(ide,maxi)]
    print("lenght max : {0}\nImage id : {1}".format(total_maxi,ide_max))
    print("number of bad images : {0}".format(len(list_max)))
    return list_max
    
#list_max= Find_max_length()
    
    
    
def Create_new_pickle_data():
    new_train_images = np.zeros((10000, 32, 32), dtype=np.float32)
    lis = []
    for ide in range(10000):
        ph = "\rProgression: {0} % ".format(round(float(100*ide)/float(10000-1),3))
        sys.stdout.write(ph)
        sys.stdout.flush()
        im = decomposition(ide)
        if type(im) == int:
            new_train_images[ide,:,:] = np.zeros((32,32), dtype = np.uint8)
            lis += [ide]
        else:
            new_train_images[ide,:,:] = im
        
    return (new_train_images,lis)



(new_train_images,lis) = Create_new_pickle_data()
print("{0} error !".format(len(lis)))
with open(directory+'/input/new_process_test_images.pkl',"wb") as file:
    pickle.dump(new_train_images,file)
