# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:11:00 2022

@author: Cornelius
"""

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os

path="C:/Users/Cornelius/Downloads"
os.chdir(path)
#os.chdir('C:\Users\Cornelius\Downloads')  
data=pd.read_csv('Flickr_nature_2020_2022.csv',encoding = "ISO-8859-1")
# initialize an axis
fig, ax = plt.subplots(figsize=(16,6))
# plot map on axis
countries = gpd.read_file(  
     gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey",ax=ax)
# plot points
data.plot(x="longitude", y="latitude", kind="scatter",c="residfaves",colormap="YlOrRd",ax=ax,s=.02)


plt.savefig("testresid.png")