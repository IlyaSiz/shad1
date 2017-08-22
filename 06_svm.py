# -*- coding: utf-8 -*-
from sklearn.svm import SVC
import pandas as pd

data = pd.read_csv('svm/svm-data.csv', header=None)
y = data[0]
X = data.drop(0, axis=1)

# SVC – машина опорных векторов
svc = SVC(kernel='linear', C=100000, random_state=241)
svc.fit(X=X, y=y)
X_values = X.values

# print svc.support_vectors_
for i, x in enumerate(X_values):
    if x in svc.support_vectors_:
        print i+1
# Опорные объекты: 4 5 10
