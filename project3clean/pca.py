
from sklearn.preprocessing import StandardScaler

import pandas as pd
from pandas import *

import numpy as np


from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA


import matplotlib.pyplot as plt

import seaborn as sns; sns.set()


df = pd.read_csv('HSData5.csv')


print(df)


#standardizing data


features = ['satMath', 'satWrt', 'Graduation', 'APScoreQualified', 'ps_enroll_male','attendance', 'enrollment']
# Separating out the features
x = df.loc[:, features].values


# Separating out the target
y = df.loc[:,['borough']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)



pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)




principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

loadings = pd.DataFrame(pca.components_.T, columns=['PC1','PC2'], index=features)

finalDf = pd.concat([principalDf, df[['borough']]], axis = 1)




# ####################################################################################






# K MEANS ELBOW METHOD CALCULATION AND GRAPH


distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
  
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(principalComponents)
    kmeanModel.fit(principalComponents)
  
    distortions.append(sum(np.min(cdist(principalComponents, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / principalComponents.shape[0])
    inertias.append(kmeanModel.inertia_)
  
    mapping1[k] = sum(np.min(cdist(principalComponents, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / principalComponents.shape[0]
    mapping2[k] = kmeanModel.inertia_

for key, val in mapping1.items():
    print(f'{key} : {val}')


plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

# K MEANS CALCULATION WITH K=3, GIVEN FROM THE CHART

kmeans = KMeans(n_clusters = 3)
kmeans.fit(principalComponents)
y_kmeans = kmeans.predict(principalComponents)


print(K)

print(type(kmeans))

print(kmeans)
print(y_kmeans)




print(principalComponents)

print(principalDf)

print(finalDf)

print(loadings)

#scree plot

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Manhattan', 'bronx', 'brooklyn', 'queens', 'statenisland']
colors = ['r', 'g', 'b', 'c', 'm']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['borough'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


