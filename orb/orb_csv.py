# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:26:27 2019

@author: Savitoj
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import glob
import csv
import time


def cluster_features(img_descs, cluster_model):

    n_clusters = cluster_model.n_clusters
    training_descs = []
    for i in range(len(img_descs)):
        if img_descs[i] is not None:
            training_descs.append(img_descs[i])
    #training_descs = [img_descs[i] for i in range(len(img_descs))]
    
    all_train_descriptors = []
    for desc_list in training_descs:
        for desc in desc_list:
            all_train_descriptors.append(desc)
    
    all_train_descriptors = np.array(all_train_descriptors)

   
    #print ('%i descriptors before clustering' % all_train_descriptors.shape[0])

   
    #print ('Using clustering model %s...' % repr(cluster_model))
    #print ('Clustering on training set to get codebook of %i words' % n_clusters)

    
    cluster_model.fit(all_train_descriptors)
    #print ('done clustering. Using clustering model to generate BoW histograms for each image.')

   
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in training_descs]

   
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print ('done generating BoW histograms.')

    return X, cluster_model



orb = cv2.ORB_create(nfeatures=500) 
map_ = ['A','B','C','D','del','E','F','G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']


img = cv2.imread('../../../asl_alphabet_train/asl_alphabet_train/C/C4.jpg',0)
#img = cv2.imread('simple.jpg',0)


f = plt.figure()
f.add_subplot(1,2, 1)

edges = cv2.Canny(img,100,200)
plt.imshow(edges)
cv2.imwrite("c4edges.jpg",edges)
kp,des = orb.detectAndCompute(edges,None)
img2 = cv2.drawKeypoints(img,kp,None, flags=0)

f.add_subplot(1,2, 2)

plt.imshow(img2)
cv2.imwrite("descc4.jpg", img2)
plt.show(block=True)

'''

start = time.time()

for alpha in map_:
   
    i=0
    label = map_.index(alpha)
    img_descs=[]
    img_links= []
    for imgs in glob.glob('../../../asl_alphabet_train/asl_alphabet_train/'+alpha+'/*.jpg'):
        i+=1
        
        img = cv2.imread(imgs,0)
        edges = cv2.Canny(img,100,200)
        kp,des = orb.detectAndCompute(edges,None)
        #flattened_sign_image=des.flatten()
        img_links.append(imgs)
        #outputLine = [label] + np.array(flattened_sign_image).tolist()
        #plt.imshow(edges),plt.show()
        img_descs.append(des)
        

    X, cluster_model = cluster_features(img_descs, MiniBatchKMeans(n_clusters=150))
    X = X.tolist()
    for row in X:
        row.append(label)
     
    with open("asl_dataset_orb.csv","a") as mycsv:
        csvWriter = csv.writer(mycsv,delimiter=',')
        csvWriter.writerows(X)
    print("done " + alpha + str(time.time() - start))

#Skin masking
'''
'''
converted2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    #cv2.imshow("original",converted2)

lowerBoundary = np.array([0,40,30],dtype="uint8")
upperBoundary = np.array([43,255,254],dtype="uint8")
skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
 #cv2.imshow("masked",skinMask)
    
skinMask = cv2.medianBlur(skinMask, 5)
    
skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
'''




