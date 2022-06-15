# Imports
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_pickle("/Users/Yohan/Downloads/02466Fagprojekt-main/eksempel.pkl")

"""
data["image"] = ""
for i in range(len(data)):
    data["image"][i] = data.iloc[i]["Image path"].split("/")[-1]
"""

av_col = []
for i in data["Image path"]:
    try: 
        src_img = cv2.imread(f"{os.getcwd()}/images/{i}")
        src_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)
        average_color_row = np.average(src_img, axis=0)
        average_color = np.average(average_color_row, axis=0) 
        av_col.append(average_color)

        """
        d_img = np.ones((312,312,3), dtype=np.uint8)
        d_img[:,:] = average_color
        cv2.imshow('Source image',src_img)
        cv2.imshow('Average Color',average_color)
        cv2.waitKey(0)
        """
    except:
        continue

#plt.bar(np.arange(len(average_color)), [red_percentage, g_percentage, b_percentage] , color=["r","g","b"], edgecolor = 'black')
#plt.xlabel("RGB-colors")
#plt.ylabel("Percentage color")
#plt.title("Color density)

red_percentage = []
green_percentage = []
blue_percentage = []
for i in av_col:
    #red_percentage.append(i[0])
    #green_percentage.append(i[1])
    #blue_percentage.append(i[2])
    
    red_percentage.append(i[0]/np.sum(i))
    green_percentage.append(i[1]/np.sum(i))
    blue_percentage.append(i[2]/np.sum(i))
    
    


plt.scatter(red_percentage, data["Predicted Resid"], color="red")
plt.show()
plt.scatter(green_percentage, data["Predicted Resid"], color="green")
plt.show()
plt.scatter(blue_percentage, data["Predicted Resid"], color="blue")
plt.show()



