import cv2
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import pickle




#img = cv2.imread(r'C:\Users\Alexandre\Documents\ETS-Session-2\COMP-551\Projet 3\Image.jpg')
directory  = "C:/Users/Alexandre/Documents/ETS-Session-2/COMP-551/Projet 3"
train_images = pd.read_pickle(directory+'/input/train_images.pkl')
train_labels = pd.read_csv(directory+'/input/train_labels.csv')

def printer(ide):
    fig = plt.figure()
    plt.title('Label: {}'.format(train_labels.iloc[ide]['Category']))
    plt.imshow(train_images[ide])

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


def decomposition(ide):

    img = np.zeros((64,64,3), dtype = np.uint8)
    img_cleaned = dele(train_images[ide])
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
        lenght = max(w,h)
        if lenght > maxi:
            mx = x,y,w,h
            maxi = lenght
    x,y,w,h = mx
    
    # Output to files
#    printer(ide)
    roi=img[y:y+h,x:x+w]
#    cv2.imwrite(r'C:\Users\Alexandre\Documents\ETS-Session-2\COMP-551\Projet 3\Image_crop.jpg', roi)
#    cv2.rectangle(img,(x-1,y-1),(x+w,y+h),(200,0,0),1)
#    cv2.imwrite(r'C:\Users\Alexandre\Documents\ETS-Session-2\COMP-551\Projet 3\Image_cont.jpg', img)
    
    roi_final = dele_change(roi[:,:,0])
#    fig = plt.figure()
#    plt.imshow(roi_final)
    
    new_img = np.zeros((51,51), dtype = np.uint8)
    (x_c,y_c) = (25-int(h/2),25-int(w/2))
    new_img[x_c:x_c+h,y_c:y_c+w] = roi_final
#    fig = plt.figure()
#    plt.imshow(new_img)
    
    return new_img
    

def Find_max_length():
    total_maxi = 0
    ide_max = 0
    for ide in range(len(train_images)):
        ph = "\rProgression: {0} % ".format(round(float(100*ide)/float(len(train_images)-1),3))
        sys.stdout.write(ph)
        sys.stdout.flush()
        maxi = decomposition(ide)
        if maxi > total_maxi:
            total_maxi = maxi
            ide_max = ide
    print("lenght max : {0}\nImage id : {1}".format(total_maxi,ide_max))
    
    
def Create_new_pickle_data():
    new_train_images = np.zeros((40000, 51, 51), dtype=np.float32)
    for ide in range(40000):
        ph = "\rProgression: {0} % ".format(round(float(100*ide)/float(len(train_images)-1),3))
        sys.stdout.write(ph)
        sys.stdout.flush()
        new_train_images[ide,:,:] = decomposition(ide)
        
    return new_train_images

new_train_images = Create_new_pickle_data()
with open(directory+'/input/new_train_images.pkl',"wb") as file:
    pickle.dump(new_train_images,file)


#fig = plt.figure()
#img = plt.imread(r'C:\Users\Alexandre\Documents\ETS-Session-2\COMP-551\Projet 3\Image_crop.jpg')
#plt.imshow(img)
#fig = plt.figure()
#img = plt.imread(r'C:\Users\Alexandre\Documents\ETS-Session-2\COMP-551\Projet 3\Image_cont.jpg')
#plt.imshow(img)