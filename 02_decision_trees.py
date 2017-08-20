# -*- coding: utf-8 -*-

# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# X = np.array([[1, 2], [3, 4], [5, 6]])
# y = np.array([0, 1, 0])
# clf = DecisionTreeClassifier()
# clf.fit(X, y)
#
# importances = clf.feature_importances_
#
# print importances

import pandas
from sklearn.tree import DecisionTreeClassifier

# загрузка данных в Pandas:
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
actualData = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
actualData = actualData.dropna()
actualData['Sex'] = (actualData['Sex'] == 'male')
objects = actualData[['Pclass', 'Fare', 'Age', 'Sex']]
results = actualData['Survived']
clf = DecisionTreeClassifier(random_state=241)
clf.fit(objects, results)
importances = clf.feature_importances_
print importances
# [ 0.14000522  0.30343647  0.2560461   0.30051221]
# Fare Sex
