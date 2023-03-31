from sklearn.manifold import MDS

import pandas as pd
from pandas import *

df = pd.read_csv('HSData5.csv')


print(df)


#standardizing data


features = ['satMath', 'satWrt', 'Graduation', 'APScoreQualified', 'ps_enroll_male','attendance', 'enrollment']
# Separating out the features
x = df.loc[:, features].values


# Separating out the target
y = df.loc[:,['borough']].values

embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(x)


df2 = pd.DataFrame(data = X_transformed, columns = ['x', 'y'])

print(X_transformed)

print(df2)

df2.to_csv('out2.csv')