# -*- coding: utf-8 -*-

import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import sklearn

# загрузка данных в Pandas:
data = pandas.read_csv('wine.data', header=None)

# Pandas Dataframe to np.array
X = data.ix[:, 1:].values
y = data.ix[:, :0].values

X_scaled = sklearn.preprocessing.scale(X)

cv_scores = []
cv_scores_scaled = []
# Создаем генератор разбиений
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# kfold.get_n_splits(X)

# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

for k in range(50):
    k_value = k + 1
    knn = KNeighborsClassifier(n_neighbors=k_value)
    scores = cross_val_score(estimator=knn, X=X, y=y.ravel(), cv=kfold, scoring='accuracy')
    scores_scaled = cross_val_score(estimator=knn, X=X_scaled, y=y.ravel(), cv=kfold, scoring='accuracy')
    mean = scores.mean()
    mean_scaled = scores_scaled.mean()
    cv_scores.append(mean)
    cv_scores_scaled.append(mean_scaled)
    # print k_value, mean
max_score = max(cv_scores)
max_score_scaled = max(cv_scores_scaled)
optimal_k = cv_scores.index(max_score) + 1
optimal_k_scaled = cv_scores_scaled.index(max_score_scaled) + 1
print "The optimal number of neighbors is %d with %f" % (optimal_k, max_score)
print "The optimal number of neighbors with scaled features is %d with %f" % (optimal_k_scaled, max_score_scaled)

# The optimal number of neighbors is 1 with 0.730476
# The optimal number of neighbors is 29 with 0.977619
