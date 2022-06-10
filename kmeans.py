import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.cluster import KMeans
from collections import Counter
import cv2



imagename = "/Users/Yohan/Downloads/02466Fagprojekt-main/images/49763410506_729ce62fef_z.jpg"

img=cv2.imread(imagename)
plt.imshow(img)

#RGB
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)


img=img.reshape((img.shape[1]*img.shape[0],3))

kmeans=KMeans(n_clusters=5)
s=kmeans.fit(img)


labels=kmeans.labels_
print(labels)
labels=list(labels)

centroid=kmeans.cluster_centers_
print(centroid)


percent=[]
for i in range(len(centroid)):
  j=labels.count(i)
  j=j/(len(labels))
  percent.append(j)
print(percent)



plt.bar(range(len(centroid)), percent, color=centroid/255)
plt.xlabel("Colors")
plt.ylabel("Percentage color")
plt.show()


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

