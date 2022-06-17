# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:56:49 2022

@author: Corne
"""
from PIL import Image
import numpy as np
from skimage.feature import greycomatrix
import skimage
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

test_data="predicted_df.pkl"
data = pd.read_pickle(test_data)
data["entropy"]=""
count=0
for path in data["Image path"]:
    try:
        count+=1  #tæller op med det samme, så række nummer stadig passer selvom invalid image
        print(count)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        entropy=skimage.measure.shannon_entropy(img)

        data.iloc[count-1,3]=entropy

    except:
        pass
    

data['entropy'].replace('', np.nan, inplace=True)
data.dropna(subset=['entropy'], inplace=True)


#Gemmer ny pickle med object-coun
data.to_pickle("predicted_df.pkl")




correlation,p_value=stats.pearsonr(data["entropy"].tolist(),data["Predicted Resid"].tolist())
plt.scatter(data["entropy"],data["Predicted Resid"],color="black", alpha=1,s=.2,marker="o")
plt.title(f"Predicted Residfaves and shannon entropy, corr= {correlation:.2f}")
plt.xlabel("entropy")
plt.ylabel("Predicted Residfaves")
plt.savefig('entropy_predicted_resid.png',
            dpi=200)
plt.show()


plt.scatter(data["Predicted Resid"],data["True Resid"],color="black", alpha=1,s=.2,marker="o")
correlation,p_value=stats.pearsonr(data["Predicted Resid"].tolist(),data["True Resid"].tolist())
plt.plot([-5, 2], [-5 , 2], 'k-', color = 'r')
plt.title(f"True Residfaves and Predicted Resid, corr= {correlation:.2f}")
plt.xlabel("Predicted Resid")
plt.ylabel("True Residfaves")
plt.savefig('predicted_resid_true_resid.png',
            dpi=200)
plt.show()