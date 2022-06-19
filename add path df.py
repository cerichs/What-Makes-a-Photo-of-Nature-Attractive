# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 17:21:10 2022

@author: Corne
"""
import pandas as pd
import sys, os
from numpy.random import RandomState

#data = pd.read_pickle("C:/Users/Corne/Documents/GitHub/02466Fagprojekt/dataframesaved.pkl")
data = pd.read_pickle("C:/Users/Corne/Documents/GitHub/02466Fagprojekt/samlet_df_temp.pkl")
for i in range(0,180000):
    try:
        url = data['url_z'][i]
        data["path"][i] = f"{os.path.dirname(os.path.abspath(url.split('/')[-1] ))}/images/{url.split('/')[-1]}"  
        print(f"|{i/len(data)*100}% csv done| Currently, on file:{i} out of:{len(data)}")
    except:
        continue 


rng = RandomState(seed=20)

df = data[["residfaves", "url_z", "path", "height_z", "width_z","tags"]]
#df_train = df.sample(frac=0.8, random_state=rng)
#df_test = df.loc[~df.index.isin(df_train.index)]

df.to_pickle("samlet_df_temp1.pkl")
#df_train.to_pickle("train_df.pkl")
#df_test.to_pickle("test_df.pkl")