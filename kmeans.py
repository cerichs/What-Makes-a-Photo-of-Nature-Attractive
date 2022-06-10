import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
import cv2


def Kmeans(pictures, clusters):
    fig = plt.figure()
    
    #Indlæser billed
    img=cv2.imread(pictures)
    
    #Indlæser med RGB
    imgo=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    (h,w,c) = imgo.shape
    plt.subplot(1, 2, 1)
    plt.imshow(imgo)
    
    #Vektorisere
    img=imgo.reshape((imgo.shape[1]*imgo.shape[0],3))
    
    #Laver K-means på antallet af clusters
    kmeans=KMeans(n_clusters=clusters, random_state = 1)
    #Fitter K-means på billedet
    s=kmeans.fit(img)
    
    #Angiver antal labels på antal clusters
    labels=kmeans.labels_
    labels=list(labels)
    
    #Finder centrum af farver
    centroid=kmeans.cluster_centers_
    rgb_cols = kmeans.cluster_centers_.round(0).astype(int)
    
    
    
    #Finder procentdel af farven i billedet
    percent=[]
    for i in range(len(centroid)):
      j=labels.count(i)
      j=j/(len(labels))
      percent.append(j)
    
    #
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(centroid)), percent, color=centroid/255)
    plt.xlabel("Colors")
    plt.ylabel("Percentage color")
    plt.show()
    
    
    #Udregner euklidisk distance mellem antallet af clusters
    dists = euclidean_distances(kmeans.cluster_centers_)
    #Tager gennemsnitlig distance af antal clusters
    tri_dists = dists[np.triu_indices(clusters, 1)]
    max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
    
    return max_dist, avg_dist, min_dist


print(Kmeans("/Users/Yohan/Downloads/02466Fagprojekt-main/images/50634283257_54e4759bfb_z.jpg", 5))
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

