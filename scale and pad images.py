
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
                dim = np.shape(img)
                leftPad = round(float((640 - dim[0])) / 2)
                rightPad = round(float(640 - dim[0]) - leftPad)
                topPad = round(float((640 - dim[1])) / 2)
                bottomPad = round(float(640 - dim[1]) - topPad)
                pads = ((leftPad,rightPad),(topPad,bottomPad),(0,0))
                img=np.pad(img,pads,'constant')
                Img=Image.fromarray(np.uint8(img)).convert('RGB')
                
            if Scaling==True:
                Img = Img.resize(Scaling_dim,resample=Image.LANCZOS)
                #print("C:/Users/Corne/Documents/GitHub/02466Fagprojekt/images scaled/" + str(path.split('/')[-1]))
            Img.save("C:/Users/Corne/Documents/GitHub/02466Fagprojekt/images scaled/" + str(path.split('/')[-1]),quality=90)
            i=i+1
            #Image.fromarray(np.uint8(img)).convert('RGB').save("C:/Users/Corne/Documents/GitHub/02466Fagprojekt/images scaled/" + str(path.split('/')[-1]))
            
            if i%100==0:
                print(i)
        except:
            pass

df=pd.read_pickle("samlet_df_temp1.pkl")
preprocess(df,(224,224),padding=True,Scaling=True)
