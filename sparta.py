import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import *

data = pd.read_csv('wine.data', header=None)
X = pd.read_csv('wine.data', header=None, usecols=list(xrange(1, 14)))
y = pd.read_csv('wine.data', header=None, usecols=[0]).values.reshape(len(X), )

kf = KFold(n_splits=5, shuffle=True, random_state=42)

kMeans = list()
for k in range(1, 51):
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(X, y)
    array = cross_val_score(estimator=kn, X=X, y=y, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m)

m = max(kMeans)
indices = [i for i, j in enumerate(kMeans) if j == m]

print indices[0] + 1
print np.round(m, decimals=2)

X_scale = scale(X)

kMeans = list()
for k in range(1, 51):
    kn = KNeighborsClassifier(n_neighbors=k)
    array = cross_val_score(estimator=kn, X=X_scale, y=y, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m)

m = max(kMeans)
indices = [i for i, j in enumerate(kMeans) if j == m]

print indices[0] + 1
print np.round(m, decimals=2)