
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


###Correlation heat map
corr = temp_df.corr()
f, ax = plt.subplots(figsize=(10,10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
heatmap = sns.heatmap(corr, cmap=cmap, center=0.0, vmax=1, linewidth=1, ax=ax)
plt.show()


x=temp_df[["entropy","objects","average color red","average color green","average color blue", "average color total", "avg_dist"]]
y=temp_df["Predicted Resid"]
scaler = StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
pca = PCA()
pc = pca.fit_transform(x)
pc_df = pd.DataFrame(data = pc, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
finalDf = pd.concat([pc_df, temp_df["Predicted Resid"]], axis = 1)
finalDf.dropna(subset = ["PC1"], inplace=True)


fig = plt.figure(figsize = (8,6), dpi=250)
plt.scatter(pc[:,0], pc[:,1], c = finalDf["Predicted Resid"], cmap="hsv", s = .8, alpha=.5)
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.title('2 component PCA', fontsize = 20)
plt.colorbar()
plt.grid(visible=1, axis="both")
plt.show()


cum_sum_exp = np.cumsum(pca.explained_variance_ratio_)
plt.bar(range(0,len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_exp)), cum_sum_exp, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

df = temp_df.loc[:, ["entropy","objects","average color red","average color green","average color blue", "average color total", "avg_dist"]]

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = xs
    scaley = ys
    plt.scatter(xs ,ys, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, ["entropy","objects","average color red","average color green","average color blue", "average color total", "avg_dist"][i],  color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

#Call the function. Use only the 2 PCs.
myplot(pc[:,0:2],np.transpose(pca.components_[0:2, :]))
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
