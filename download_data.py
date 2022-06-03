
import pandas as pd
import cv2, wget
import numpy as np
import sys, os
import glob
from datetime import datetime
import random
start_time = datetime.now()


#skaber funktion der returnere ny CSV-fil med image-path

#indtager path samt antal af billeder der skal medtages

def NewCSV(path, number):
    
    
    
    #Læser alle filer i path med navn indeholdende ".csv":
    allfiles = glob.glob(path + "/*.csv")
    
    dat = []
    for i in allfiles:
        dat.append(pd.read_csv(i, sep=",", encoding='latin-1'))
    
    data = pd.concat(dat)
    
    #Sletter URL NAN-rækker
    data.dropna(subset = ["url_z"], inplace=True)
    data.drop_duplicates(subset ="url_z",
                     keep = False, inplace = True)
    data = data.reset_index(drop=True)
    
    #Tager n tilfældige billeder fra CSV:
    n = random.sample(range(0, len(data)), number)
    
    #Downloader tilgængelige billeder og gemmer paths
    for i in n: #skal bare ændres til range(len(data))
        try: ##Ikke alle url er stadig aktiv derfor try/except nødvendig
            url = data['url_z'][i]
            if url.split('/')[-1] in os.listdir(): #undlader duplicates
                continue
            else:
                res=wget.download(url)
            
        except: ##hvis url er ugyldig fjerner række fra data
            data = data.drop(index=i)
            
    
    #Skaber empty-kolonne som udfyldes iterativt med path til filer
    #data["path"] = ""
    #for i, j in zip(idx, range(len(photo_path))):
    #    data['path'][i] = photo_path[j]
    
    data["path"] = ""
    for i in range(len(data)):
        try:
            url = data['url_z'][i]
            if url.split('/')[-1] in os.listdir():
                data["path"][i] = f"{os.path.dirname(os.path.abspath(url.split('/')[-1] ))}/{url.split('/')[-1]}"  
            else:
                continue
        except:
            continue
            
    df = data[data.path != ""]
    df.to_csv("output.csv")



#eksempelvis nuværende directory og 10 billeder:
#print(NewCSV(os.getcwd(), 10))
    

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))




