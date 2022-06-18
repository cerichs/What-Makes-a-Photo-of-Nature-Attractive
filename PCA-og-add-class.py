
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score


temp_df=pd.read_pickle("predict_df_to_pca.pkl")
sns.set_theme(color_codes=True)
temp_df['average color red'] = pd.to_numeric(temp_df['average color red'])
temp_df['average color blue'] = pd.to_numeric(temp_df['average color blue'])
temp_df['average color green'] = pd.to_numeric(temp_df['average color green'])
temp_df['average color total'] = pd.to_numeric(temp_df['average color total'])
temp_df['entropy'] = pd.to_numeric(temp_df['entropy'])
temp_df['objects'] = pd.to_numeric(temp_df['objects'])
temp_df['avg_dist'] = pd.to_numeric(temp_df['avg_dist'])


classes=["very unacctractive","unattractive", "neutral", "attractive"]
Q=[-5.43106544, -0.48404679,  0.09200823,  0.57377643,  2.62637219]
temp=[]
j=0
for entries in data["Predicted Resid"]:
    for i in range(0,4,1):
        if (Q[i] < entries < Q[i+1]):
            temp.append(classes[i])
        else:
            pass
    if j%500==0:
        print(j)
    j=j+1
data["class"]=temp


x=data[["entropy","objects","average color red","average color green","average color blue", "average color total", "avg_dist"]]
y=data["class"]
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=7)
pc = pca.fit_transform(x)
pc_df = pd.DataFrame(data = pc, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
finalDf = pd.concat([pc_df], axis = 1)


fig = plt.figure(figsize = (8,6), dpi=250)
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

colors = []
for i,j,k, t in zip(data["average color red"],data["average color green"],data["average color blue"], data["average color total"]):
    c = [i*t,j*t,k*t]
    colors.append(c)



ax.scatter(finalDf.loc[:, 'PC1'], finalDf.loc[:, 'PC2'], c = np.array(colors)/255.0, s = .8, alpha=.5)
#ax.legend(classes)
ax.grid()
plt.show()

cum_sum_exp = np.cumsum(pca.explained_variance_ratio_)
plt.bar(range(0,len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_exp)), cum_sum_exp, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


"""
correlation,p_value=stats.pearsonr(temp_df["avg_dist"].tolist(),temp_df["Predicted Resid"].tolist())
correlation_matrix = np.corrcoef(temp_df["avg_dist"].tolist(), temp_df["Predicted Resid"].tolist())
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2

r2_score(temp_df["avg_dist"].tolist(),temp_df["Predicted Resid"].tolist())
ax = sns.regplot(x="avg_dist", y="Predicted Resid", data=temp_df, scatter_kws={"color": "olive",  "s": 5},line_kws={"color": "black"})
plt.title(f"Predicted Residfaves and color heterogeneity, corr= {correlation:.2f}, r^2={r_squared}")
plt.xlabel("avg_dist")
plt.ylabel("Predicted Residfaves")
plt.savefig('avgdist_predicted_resid.png',       
            dpi=200)
plt.show()


correlation,p_value=stats.pearsonr(temp_df["entropy"].tolist(),temp_df["Predicted Resid"].tolist())
correlation_matrix = np.corrcoef(temp_df["entropy"].tolist(), temp_df["Predicted Resid"].tolist())
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
ax = sns.regplot(x='entropy', y="Predicted Resid", data=temp_df, scatter_kws={"color": "grey",  "s": 5},line_kws={"color": "black"})
plt.title(f"Predicted Residfaves and shannon entropy, corr= {correlation:.2f}, r^2={r_squared}")
plt.xlabel("entropy")
plt.ylabel("Predicted Residfaves")
plt.savefig('entropy_predicted_resid.png',
            dpi=200)
plt.show()



correlation,p_value=stats.pearsonr(temp_df["objects"].tolist(),temp_df["Predicted Resid"].tolist())
correlation_matrix = np.corrcoef(temp_df["objects"].tolist(), temp_df["Predicted Resid"].tolist())
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
ax = sns.regplot(x='objects', y="Predicted Resid", data=temp_df, scatter_kws={"color": "peru",  "s": 5},line_kws={"color": "black"})
plt.title(f"Predicted Residfaves and counted object, corr= {correlation:.2f}, r^2={r_squared}")
plt.xlabel("Counted objects")
plt.ylabel("Predicted Residfaves")
plt.savefig('count_object_predicted_resid.png',
            dpi=200)
plt.show()





temp_df['average color red']=temp_df['average color red']/temp_df['average color total']
temp_df['average color green']=temp_df['average color green']/temp_df['average color total']
temp_df['average color blue']=temp_df['average color blue']/temp_df['average color total']

temp_df = pd.read_pickle("/Users/Yohan/Downloads/02466Fagprojekt-main/predict_df_to_pca.pkl")


temp_df['average color red'] = pd.to_numeric(temp_df['average color red'])
temp_df['average color blue'] = pd.to_numeric(temp_df['average color blue'])
temp_df['average color green'] = pd.to_numeric(temp_df['average color green'])
temp_df['average color total'] = pd.to_numeric(temp_df['average color total'])
    
correlationr,p_valuer=stats.pearsonr(temp_df['average color red'],temp_df["Predicted Resid"].tolist())
correlationg,p_valueg=stats.pearsonr(temp_df['average color green'],temp_df["Predicted Resid"].tolist())
correlationb,p_valueb=stats.pearsonr(temp_df['average color blue'],temp_df["Predicted Resid"].tolist())
correlationt,p_valuet=stats.pearsonr(temp_df['average color total'],temp_df["Predicted Resid"].tolist())



correlation_matrix = np.corrcoef(temp_df['average color red'].tolist(), temp_df["Predicted Resid"].tolist())
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
ax = sns.regplot(x='average color red', y="Predicted Resid", data=temp_df, scatter_kws={"color": "red",  "s": 5},line_kws={"color": "black"})
plt.xlabel("Red percentage")
plt.ylabel("Predicted Resid")
plt.title(f"Color density, corr= {correlationr:.2f}, r^2={r_squared}")
plt.savefig('color_density_red_percentage.png',
            dpi=200)
plt.show()

correlation_matrix = np.corrcoef(temp_df['average color green'].tolist(), temp_df["Predicted Resid"].tolist())
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
ax = sns.regplot(x='average color green', y="Predicted Resid", data=temp_df, scatter_kws={"color": "green",  "s": 5},line_kws={"color": "black"})
plt.xlabel("Green percentage")
plt.ylabel("Predicted Resid")
plt.title(f"Color density, corr= {correlationg:.2f}, r^2={r_squared}")
plt.savefig('color_density_green_percentage.png',
            dpi=200)
plt.show()

correlation_matrix = np.corrcoef(temp_df['average color blue'].tolist(), temp_df["Predicted Resid"].tolist())
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
ax = sns.regplot(x='average color blue', y="Predicted Resid", data=temp_df, scatter_kws={"color": "blue",  "s": 5},line_kws={"color": "black"})
plt.xlabel("Blue percentage")
plt.ylabel("Predicted Resid")
plt.title(f"Color density, corr={correlationb:.2f}, r^2={r_squared}")
plt.savefig('color_density_blue_percentage.png',
            dpi=200)
plt.show()




colors = []
for i,j,k, t in zip(temp_df["average color red"],temp_df["average color green"],temp_df["average color blue"], temp_df["average color total"]):
    c = [i*t,j*t,k*t]
    colors.append(c)

correlation_matrix = np.corrcoef(temp_df['average color total'].tolist(), temp_df["Predicted Resid"].tolist())
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
ax = sns.regplot(x='average color total', y="Predicted Resid", data=temp_df, scatter_kws={"color": np.array(colors)/255.0, "s": 5},line_kws={"color": "black"})
plt.xlabel("Mean Color")
plt.ylabel("Predicted Resid")
plt.title(f"Color density, corr= {correlationt:.2f}, r^2={r_squared}")
plt.savefig('color_density_color_percentage.png',
            dpi=200)
plt.show()

"""
