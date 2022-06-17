# -*- coding: utf-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

test_data="predicted_df.pkl"
data = pd.read_pickle(test_data)
data["objects"]=""
count=0
for path in data["Image path"]:
    try:
        count+=1  #tæller op med det samme, så række nummer stadig passer selvom invalid image
        print(count)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(img, (15, 15), 0)
        canny = cv2.Canny(blur, 100, 150, 3)

        
        (cnt, hierarchy) = cv2.findContours(
            canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        data["objects"].iloc[count-1]=len(cnt)
    except:
        pass


data['objects'].replace('', np.nan, inplace=True)
data.dropna(subset=['objects'], inplace=True)


#Gemmer ny pickle med object-count
data.to_pickle("predicted_df.pkl")





plt.scatter(data["objects"],data["Predicted Resid"],color="black", alpha=1,s=.2,marker="o")
correlation,p_value=stats.pearsonr(data["objects"].tolist(),data["Predicted Resid"].tolist())
plt.title(f"Predicted Residfaves and counted object, corr= {correlation:.2f}")
plt.xlabel("Counted objects")
plt.ylabel("True Residfaves")
plt.savefig('count_object_predicted_resid.png',
            dpi=200)
plt.show()
plt.hist(data["objects"],bins=25)
plt.xlabel("Counted Objects")
plt.savefig('count_object_hist.png',
            dpi=200)
plt.show()

