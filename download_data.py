# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:11:00 2022

@author: Cornelius
"""

import pandas as pd
import cv2, wget
import numpy as np
import re
import sys, os
import shutil
import glob
from sys import exit
from datetime import datetime
start_time = datetime.now()

#Angiver pathen til file(r):
path = os.getcwd()

#Læser alle filer i path med navn indeholdende ".csv":
allfiles = glob.glob(path + "/*.csv")

dat = []
for i in allfiles:
    dat.append(pd.read_csv(i, sep=",", encoding='latin-1'))

data = pd.concat(dat) 

#Sletter URL NAN-rækker
data.dropna(subset = ["url_z"], inplace=True)
data=data.reset_index(drop=True)

#Downloader tilgængelige billeder og gemmer paths
photo_path = []
idx = []
for i in range(0, 100):
    try: ##Ikke alle url er stadig aktiv derfor try/except nødvendig
        url=data['url_z'][i]
        if url.split('/')[-1] in os.listdir(): #undlader duplicates
            photo_path.append(url.split('/')[-1])
            idx.append(i)
            continue
        else:
            res=wget.download(url)
            idx.append(i)
            photo_path.append(res)
            img_arr = cv2.imread(res)
        
    except: ##hvis url er ugyldig fjerner række fra data
        data=data.drop(index=i)
        

#Skaber empty-kolonne som udfyldes iterativt med path til filer
data["path"] = " "
for i, j in zip(range(len(idx)), range(len(photo_path))):
    data['path'][i] = photo_path[j]


    



end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))