import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
import cv2
import os
import seaborn as sns
from sklearn.metrics import mean_squared_error
import scipy

#from download_data import *

#NewCSV("dataframesaved.pkl", 0.8, 100)


def Kmeans(test_data, K):
    
    try:
    #Læser fil med angivet filnavn:
        data = pd.read_csv(test_data, sep=",", encoding='latin-1')
    except:
        data = pd.read_pickle(test_data)
    
    """
    billeder  = []
    for i in test_data["Image path"]:
        billeder.append(i)
    """
    
    img = []
    for path in data["Image path"]:
        try:
            #Indlæser billeder
            img1=cv2.imread(path)
            #Indlæser med RGB
            imgo=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
            #(h,w,c) = imgo.shape
            #plt.subplot(1, 2, 1)
            #plt.axis("off")
            #plt.imshow(imgo)
        
            #Vektorisere
            img.append(imgo.reshape((imgo.shape[1]*imgo.shape[0],3)))
        except:
            continue
    
    
    max_dist=[]
    avg_dist=[]
    min_dist=[]
    
    data["avg_dist"] = ""
    for i in img:
        #Laver K-means på antallet af clusters
        kmeans=KMeans(n_clusters=K, random_state = 1)
        #Fitter K-means på billedet
        s=kmeans.fit(i)
        
        #Angiver antal labels på antal clusters
        labels=kmeans.labels_
        labels=list(labels)
        
        #Finder centrum af farver
        centroid=kmeans.cluster_centers_
        rgb_cols = kmeans.cluster_centers_.round(0).astype(int)
        
        
        
        #Finder procentdel af farven i billedet
        """
        percent=[]
        for c in range(len(centroid)):
          j=labels.count(c)
          j=j/(len(labels))
          percent.append(j)
        """
        
        
            #plt.subplot(1, 2, 2)
            #plt.bar(np.arange(len(centroid)), percent, color=centroid/255, edgecolor = 'black')
            #plt.xlabel("Colors")
            #plt.ylabel("Percentage color")
            #plt.show()
        
    
        #Udregner euklidisk distance mellem antallet af clusters
        dists = euclidean_distances(kmeans.cluster_centers_)
        #Tager gennemsnitlig distance af antal clusters
        tri_dists = dists[np.triu_indices(K, 1)]
        
        data["avg_dist"] = tri_dists.mean()
        
        max_dist.append(tri_dists.max())
        avg_dist.append(tri_dists.mean())
        min_dist.append(tri_dists.min())


    x=data["avg_dist"]
    y=data["Predicted Resid"]
    
    _ = sns.scatterplot(data=data, x="avg_dist", y="Predicted Resid",
                        color="black", alpha=0.5)
    
    linear_regression = LinearRegression()
    linear_regression.fit(x, y)
    target_predicted = linear_regression.predict(x)
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    mse = mean_squared_error(y, target_predicted)
    
    ax = sns.scatterplot(data=data, x="avg_dist", y="Predicted Resid",
                         color="black", alpha=0.5)
    ax.plot(x, target_predicted)
    _ = ax.set_title(f"Mean squared error = {mse:.2f},  R-squared: {res.rvalue**2:.6f} ")
    
    
    return max_dist, avg_dist, min_dist


print(Kmeans("predicted_df.pkl", 5))



### optimal K
"""
md=[]
for i in range(1,21):
  kmeans=KMeans(n_clusters=i)
  kmeans.fit(img)
  o=kmeans.inertia_
  md.append(o)
print(md)


plt.plot(list(np.arange(1,21)),md)
plt.show()
"""

