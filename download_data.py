# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:11:00 2022

@author: Cornelius
"""

import pandas as pd
import cv2, wget
data=pd.read_csv('Flickr_nature_2020_2022.csv',encoding = "ISO-8859-1")
for i in range(len(data)):
    try: ##Ikke alle url er stadig aktiv derfor try/except nødvendig
        url=data['url_l'][i]
        res=wget.download(url)
        img_arr = cv2.imread(res)
    except: ##hvis url er ugyldig fjerner række fra data
        data=data.drop(index=i)