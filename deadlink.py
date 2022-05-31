import requests
import pandas as pd
import cv2, wget
import numpy as np
import sys, os
import glob
from datetime import datetime
import random

def link_checker(link):
    try:
        #GET request
        req = requests.get(link)
            
        dead = 0
        #check status-code
        if req.status_code in [400,404,403,408,409,501,502,503]:
            dead += 1

    #Exception
    except requests.exceptions.RequestException as e:
        # print link with Errs
        raise SystemExit(f"{link}: Somthing wrong \nErr: {e}")
    return dead

data = pd.read_csv("Flickr_nature_2020_2022.csv", sep=",", encoding='latin-1')
urls = []
for i in range(len(data)):
    urls.append(data.iloc[i]["url_z"])
    
for url in urls:
    link_checker(url)