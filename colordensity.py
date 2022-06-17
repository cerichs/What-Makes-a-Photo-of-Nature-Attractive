# Imports
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



test_data="predicted_df.pkl"
data = pd.read_pickle(test_data)

"""
data["image"] = ""
for i in range(len(data)):
    data["image"][i] = data.iloc[i]["Image path"].split("/")[-1]
"""

av_col = []


temp_df=pd.DataFrame()
temp_df["Predicted Resid"]=data["Predicted Resid"]



data["average color red"]=""
data["average color green"]=""
data["average color blue"]=""
data["average color total"]=""

count=0
for i in data["Image path"]:
    try: 
        count+=1  #tæller op med det samme, så række nummer stadig passer selvom invalid image
        src_img = cv2.imread(i)
        src_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)
        average_color_row = np.average(src_img, axis=0)
        average_color = np.average(average_color_row, axis=0) 

        data["average color red"].iloc[count-1]=average_color[0]
        data["average color green"].iloc[count-1]=average_color[1]
        data["average color blue"].iloc[count-1]=average_color[2]
        data["average color total"].iloc[count-1]=np.sum(average_color)

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

data['average color red'].replace('', np.nan, inplace=True)
data.dropna(subset=['average color red'], inplace=True)
data['average color green'].replace('', np.nan, inplace=True)
data.dropna(subset=['average color green'], inplace=True)
data['average color blue'].replace('', np.nan, inplace=True)
data.dropna(subset=['average color blue'], inplace=True)
data['average color total'].replace('', np.nan, inplace=True)
data.dropna(subset=['average color total'], inplace=True)



#Gemmer ny pickle med object-coun
data.to_pickle("predicted_df.pkl")

    
temp_df['average color red']=temp_df['average color red']/temp_df['average color total']
temp_df['average color green']=temp_df['average color green']/temp_df['average color total']
temp_df['average color blue']=temp_df['average color blue']/temp_df['average color total']



    
correlationr,p_valuer=stats.pearsonr(temp_df['average color red'],temp_df["Predicted Resid"].tolist())
correlationg,p_valuer=stats.pearsonr(temp_df['average color green'],temp_df["Predicted Resid"].tolist())
correlationb,p_valuer=stats.pearsonr(temp_df['average color blue'],temp_df["Predicted Resid"].tolist())



plt.scatter(temp_df['average color red'], temp_df["Predicted Resid"], color="red")
plt.xlabel("Red percentage")
plt.ylabel("Predicted Resid")
plt.title(f"Color density, corr={correlationr}")
plt.savefig('color_density_red_percentage.png',
            dpi=200)
plt.show()
plt.scatter(temp_df['average color green'], temp_df["Predicted Resid"], color="green")
plt.xlabel("Green percentage")
plt.ylabel("Predicted Resid")
plt.title(f"Color density, corr={correlationg}")
plt.savefig('color_density_green_percentage.png',
            dpi=200)
plt.show()
plt.scatter(temp_df['average color blue'], temp_df["Predicted Resid"], color="blue")
plt.xlabel("Blue percentage")
plt.ylabel("Predicted Resid")
plt.title(f"Color density, corr={correlationb}")
plt.savefig('color_density_blue_percentage.png',
            dpi=200)
plt.show()



