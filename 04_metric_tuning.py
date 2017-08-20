# -*- coding: utf-8 -*-

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import sklearn
from sklearn import datasets

dataset = datasets.load_boston()
X = dataset.data
y = dataset.target
X = sklearn.preprocessing.scale(X)
# y = sklearn.preprocessing.scale(y)

cv_scores = []
p_list = []
# Создаем генератор разбиений
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for i, p in enumerate(np.linspace(start=1, stop=10, num=200)):
    knr = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
    scores = cross_val_score(estimator=knr, X=X, y=y, cv=kfold, scoring='neg_mean_squared_error')
    mean = scores.mean()
    cv_scores.append(mean)
    p_list.append(p)
    # print p, mean
max_score = max(cv_scores)
max_p = p_list[cv_scores.index(max_score)]
print max_p, max_score
# максимальная отрицательная квадратическая ошибка - при p = 1
