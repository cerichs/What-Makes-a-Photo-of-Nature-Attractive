
import pandas as pd
import cv2, wget
import numpy as np
import sys, os
import glob
from datetime import datetime
import random
from numpy.random import RandomState
start_time = datetime.now()


#skaber funktion der returnere ny CSV-fil med image-path

#indtager path samt antal af billeder der skal medtages

def NewCSV(filename, trainsize, number):

    try:
    #Læser fil med angivet filnavn:
        data = pd.read_csv(filename, sep=",", encoding='latin-1')
    except:
        data = pd.read_pickle(filename)

    #Sletter URL NAN-rækker
    data.dropna(subset = ["url_z"], inplace=True)
    data.drop_duplicates(subset ="url_z",
                     keep = False, inplace = True)
    data = data.reset_index(drop=True)
    
    
    
    """    
    #Tager n tilfældige billeder fra CSV:
    n = random.sample(range(0, len(data)), number)
    
    pathout = f"{os.getcwd()}/images"
    
    #Downloader tilgængelige billeder og gemmer paths
    for i in n: #skal bare ændres til range(len(data))
        print(f"|{(len(os.listdir(pathout)))/number*100}% download done| Currently, on file:{(len(os.listdir(pathout)))} out of:{number}")
        try: ##Ikke alle url er stadig aktiv derfor try/except nødvendig
            url = data['url_z'][i]
            if url.split('/')[-1] in os.listdir(): #undlader duplicates
                continue
            else:
                res=wget.download(url, out = pathout)
        except: ##hvis url er ugyldig fjerner række fra data
            data = data.drop(index=i)   
    """
    
    data["path"] = ""
    for i in range(len(data)):
        try:
            url = data['url_z'][i]
            data["path"][i] = f"{os.path.dirname(os.path.abspath(url.split('/')[-1] ))}/images/{url.split('/')[-1]}"  
            print(f"|{i/len(data)*100}% csv done| Currently, on file:{i} out of:{len(data)}")
        except:
            continue 
    
    df = data[data.path != ""]
    df.to_csv("output.csv")
    
    rng = RandomState(seed=20)

    df = df[["residfaves", "url_z", "path", "height_z", "width_z"]]
    df_train = df.sample(frac=trainsize, random_state=rng)
    df_test = df.loc[~df.index.isin(df_train.index)]

    df_train.to_csv("train.csv")
    df_test.to_csv("test.csv")
    




#eksempelvis flickr-fil med 80% trainsize og 10 billeder:
#print(NewCSV("Flickr_nature_2020_2022.csv", 0.8, 179050))
#number=179050
print(NewCSV("dataframesaved.pkl", 0.8, 179050))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))




