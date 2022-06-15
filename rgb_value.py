# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:23:01 2022

@author: natas
"""


import os
os.getcwd()
os.chdir("C:/Users/natas/OneDrive/Skrivebord/FagTest")


from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
import cv2

import seaborn as sns
from sklearn.metrics import mean_squared_error
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.image as mpimg


data =pd.read_pickle("predicted_df.pkl")

def most_common_used_color(img): 
    #Dimensions of image
    h,w=img.size
    
    # Initialize Variable
    r_total = 0
    g_total = 0
    b_total = 0
 
    count = 0
 
    # Iterate through each pixel
    for x in range(0, w):
        for y in range(0, h):
            # r,g,b value of pixel
            r, g, b = img.getpixel((x, y))
 
            r_total += r
            g_total += g
            b_total += b
            count += 1
 
    return (r_total/count, g_total/count, b_total/count)

data["red"]=""
data["green"]=""
data["blue"]=""
count=0
for path in data["Image path"]:
    try:
        count+=1
        Img=Image.open(path)
        i=np.asarray(Img, np.int32)
        h,w=i.size
        r,g,b=most_common_used_color(img)
        #data.iloc
        data["red"]=r
        data["green"]=g
        data["blue"]=b
    except:
        pass
        
    

print(most_common_used_color("predicted_df.pkl"))



# Read Image
img = Image.open(r'C:/Users/natas/OneDrive/Skrivebord/FagTest\datapoint_2.jpg')
 
# Convert Image into RGB
img = img.convert('RGB')
 
# call function
common_color = most_common_used_color(img)
 
print(common_color)
# Output is (R, G, B)
"""






        
        
        
        
        
        