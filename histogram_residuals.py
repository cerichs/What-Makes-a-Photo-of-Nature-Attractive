import pandas as pd
import cv2, wget
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

path =  os.getcwd()

data = pd.read_csv(f"{path}/Flickr_nature_2020_2022.csv", sep=",", encoding='latin-1')
data.dropna(subset = ["url_z"], inplace=True)
data.drop_duplicates(subset ="url_z",keep = False, inplace = True)
data = data.reset_index(drop=True)




# seaborn histogram density plot
sns.distplot(data['residfaves'], color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2}, bins = int(180/5))

plt.title('Histogram of attractiveness-score')
plt.xlabel('Residfaves')
plt.ylabel('Density')
plt.show()


# matplotlib histogram
plt.hist(data['residfaves'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))
# Add labels
plt.title('Histogram of attractiveness-score')
plt.xlabel('Residfaves')
plt.ylabel('Density')
plt.show()

print("")
print("")
print("Extreme-points [min;max]:   ", "[", min(data['residfaves']), " ; ", max(data['residfaves']), "]")

#############################
############################

#Kvantificerer residfaves

model = LinearRegression() #fitter linear-model til logviews og logfaves
model.fit(np.array(data["logview"]).reshape((-1, 1)), data["logfaves"])
calculated = model.predict(np.array(data["logview"]).reshape((-1, 1))) #fitter den bedst mulige lineære linje (regression)

#Udregner residualer for hvert punkt (residfaves) som er gennemsnitlig afstand fra datapunkt til bedst fitted lineære-linje
residuals = []
for i, j in zip(data["logfaves"], calculated):
    residuals.append(i-j)



#Plotter residualer mellem logviews og logfaves fra bedst fitted linje

linear_model = ols('logfaves ~ logview',
                   data=data).fit()

# display model summary
print(linear_model.summary())

# modify figure size
fig = plt.figure(figsize=(14, 8))

# creating regression plots
fig = sm.graphics.plot_regress_exog(linear_model,
                                    'logview',
                                    fig=fig)

print("")
print("")
print("Residfaves:   ", residuals)

