# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:12:05 2022

@author: Corne
"""

import pandas as pd
from matplotlib import pyplot as plt
train_df=pd.read_pickle("train_df.pkl")
test_df=pd.read_pickle("test_df.pkl")
total_df=pd.read_pickle("dataframesaved.pkl")

dataframes=[train_df,test_df]
j=0
classes=['not attractive','attractive', 'medium attractive', 'very attractive']
Q=[-5.43106544, -0.48404679,  0.09200823,  0.57377643,  2.62637219]

for data in dataframes:
    temp=[]
    for entries in data["residfaves"]:
        for i in range(0,4,1):
            if (Q[i] < entries < Q[i+1]):
                temp.append(classes[i])
            else:
                pass
        if j%500==0:
            print(j)
        j=j+1
    data["class"]=temp

train_df.to_pickle("train_df_class_str7.pkl")
test_df.to_pickle("test_df_class_str7.pkl")


plt.hist(total_df["residfaves"],bins=25)
Q=[-5.43106544, -0.48404679,  0.09200823,  0.57377643,  2.62637219]

plt.axvline(Q[0],color="red")
plt.axvline(Q[1],color="red")
plt.axvline(Q[2],color="red")
plt.axvline(Q[3],color="red")
plt.axvline(Q[4],color="red")
plt.title("Density of residfaves with quartiles")
plt.savefig('Density_quartiles.png',dpi=200)
plt.show()

plt.hist(temp)
plt.title("Classification classes")
plt.savefig('Classification_class_hist.png',dpi=200)