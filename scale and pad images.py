
from PIL import Image
import pandas as pd
import numpy as np


def preprocess(dataframe,Scaling_dim,padding=False,Scaling=False):
    i=0
    for path in df["path"]:
        try:
            Img = Image.open(path)
            if padding==True:
                img = np.asarray(Img, np.int)
                dimensions = np.shape(img)
                left = round(float((640 - dimensions[0])) / 2)
                right = round(float(640 - dimensions[0]) - left)
                top = round(float((640 - dimensions[1])) / 2)
                bottom = round(float(640 - dimensions[1]) - top)
                pads = ((left,right),(top,bottom),(0,0))
                img=np.pad(img,pads,'constant')
                Img=Image.fromarray(np.uint8(img)).convert('RGB')
                
            if Scaling==True:
                Img = Img.resize(Scaling_dim,resample=Image.LANCZOS)
                #print("C:/Users/Corne/Documents/GitHub/02466Fagprojekt/images scaled/" + str(path.split('/')[-1]))
            Img.save("C:/Users/Corne/Documents/GitHub/02466Fagprojekt/images scaled non pad/" + str(path.split('/')[-1]),quality=90)
            i=i+1
            #Image.fromarray(np.uint8(img)).convert('RGB').save("C:/Users/Corne/Documents/GitHub/02466Fagprojekt/images scaled/" + str(path.split('/')[-1]))
            
            if i%100==0:
                print(i)
        except:
            pass

df=pd.read_pickle("samlet_df_temp1.pkl")
preprocess(df,(224,224),padding=False,Scaling=True)
