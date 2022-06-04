# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 17:37:57 2022

@author: Cornelius
"""

from os import scandir
import pandas as pd
data = pd.read_csv("Flickr_nature_2020_2022.csv", sep=",", encoding='latin-1')
path= "C:/Users/Cornelius/Documents/GitHub/02466Fagprojekt/images"

obj=scandir(path) #laver iterable objekt over directory

# Fjerner nan url

data.dropna(subset = ["url_z"], inplace=True)
data.drop_duplicates(subset ="url_z",keep = False, inplace = True)
data = data.reset_index(drop=True)

dictio={}

#laver dictionary af url_z kolonnen i data
for i in range(len(data)):
    try:
        dictio[data['url_z'][i].split('/')[-1]]=i
    except:
        pass

#iterere over directory og laver lookup i vores dictionary, så kun de billeder der ikke er downloadet
#forbliver i dictionary
for entry in obj:
    if entry.is_file():
        dictio.pop(entry.name)

#fjerner de resterende entries fra dictio, da disse er billeder vi ikke har downloadet
for entries in dictio.values():
    data=data.drop(index=entries)

#gemmer dataframe
data.to_pickle("dataframesaved.pkl")