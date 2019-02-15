# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import cv2
import os
import time
import glob
import csv
from skimage.morphology import reconstruction

def kane(row, fill):
    width=0
    leftmax=0
    leftupdated=False
    rightmax=0
    rightupdated=False
    i=10
    j=190
    while i<200 and j>=0:
        if fill[row][i]==0 and leftupdated==False:
            leftmax=i
            #print(leftmax)
            leftupdated=True
        if fill[row][j]==0 and rightupdated==False:
            rightmax=j
            #print(rightmax)
            rightupdated=True
        i+=1
        j-=1
    width=abs(rightmax-leftmax)
    return width

initial_width=200

files = []
for file1 in glob.glob(r'C:\Users\Savitoj\Desktop\misc\python-asl\asl_alphabet_train\asl_alphabet_train\*'):
    for file2 in glob.glob(file1+"\*.jpg"):
        files.append(file2)

final_csv = []
start = time.time()
for i in range(len(files)):
    
    img_grayscale=cv2.imread(files[i],cv2.IMREAD_GRAYSCALE)
    ret,thresh1 = cv2.threshold(img_grayscale,105,200,cv2.THRESH_BINARY)
    imgproc = thresh1
    seed = np.copy(thresh1)
    seed[1:-1, 1:-1] = thresh1.max()
    mask=thresh1
    fill=reconstruction(seed,mask,method="erosion")
    locations=0
    
    for row in range(199, 0, -1):
        current_width = kane(row, fill)
        if current_width-initial_width >=abs(2):
            locations=(row)
            initial_width=200
            break
        else:
            initial_width=current_width
    
    for k in range(locations, 200):
        for j in range(200):
            fill[k][j] = 1
    
    new_img = cv2.resize(fill, (20,20)).reshape(400, 1).ravel()
    
    with open('asl_dataset.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(new_img)
    
    print(str(i) + "==" + str(time.time() - start))