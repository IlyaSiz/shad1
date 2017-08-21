# -*- coding: utf-8 -*-
from pandas import read_csv, DataFrame
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from sklearn import model_selection, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import pylab as pl

data = read_csv('titanic/train.csv')

cat_describe = data.describe(include=[object])
# print cat_describe['Embarked']['top']

# plot1 = data.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)
# pl.show(plot1)
# fig, axes = pl.subplots(ncols=2)
# plot2 = data.pivot_table('PassengerId', ['SibSp'], 'Survived', 'count').plot(ax=axes[0], title='SibSp')
# plot3 = data.pivot_table('PassengerId', ['Parch'], 'Survived', 'count').plot(ax=axes[1], title='Parch')
# pl.show(plot2)

data['Age'] = data['Age'].median()
data['Embarked'] = data['Embarked'].fillna(cat_describe['Embarked']['top'])

data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = data['Survived']
train = data.drop(['Survived'], axis=1)

categorical_columns = [c for c in train.columns if train[c].dtype.name == 'object']
numerical_columns = [c for c in train.columns if train[c].dtype.name != 'object']
print categorical_columns, numerical_columns

# Закодировать список с фиксированными значениями можно с помощью объекта LabelEncoder()
label = LabelEncoder()
dicts = {}
# задаем список значений для кодирования
label.fit(train['Sex'].drop_duplicates())
dicts['Sex'] = list(label.classes_)
# заменяем значения из списка кодами закодированных элементов
train['Sex'] = label.transform(train['Sex'])

# label.fit(data['Embarked'].drop_duplicates())
# dicts['Embarked'] = list(label.classes_)
# data['Embarked'] = label.transform(data['Embarked'])
train = pd.get_dummies(train, columns=['Embarked'])

# Аналогично для тестовых данных
test = read_csv('titanic/test.csv')
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test_cat_describe = test.describe(include=[object])

test['Embarked'] = test['Embarked'].fillna(test_cat_describe['Embarked']['top'])
test['Sex'] = label.transform(test['Sex'])
# result = DataFrame(test.PassengerId)
# print result
test = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
test = pd.get_dummies(test, columns=['Embarked'])

# Начинаем обучение
# количество подвыборок для валидации
kfold = 5
# список для записи результатов кросс валидации разных алгоритмов
cv_map = {}

train_x, test_x, train_y, test_y = model_selection.train_test_split(train, y, test_size=0.25)

# в параметре передаем кол-во деревьев
model_rfc = RandomForestClassifier(n_estimators=70)
# в параметре передаем кол-во соседей
model_knc = KNeighborsClassifier(n_neighbors=18)
model_lr = LogisticRegression(penalty='l1', tol=0.01)
# по умолчанию kernek='rbf'
model_svc = svm.SVC()

scores = model_selection.cross_val_score(model_rfc, train, y, cv=kfold)
cv_map['RandomForestClassifier'] = scores.mean()
scores = model_selection.cross_val_score(model_knc, train, y, cv=kfold)
cv_map['KNeighborsClassifier'] = scores.mean()
scores = model_selection.cross_val_score(model_lr, train, y, cv=kfold)
cv_map['LogisticRegression'] = scores.mean()
scores = model_selection.cross_val_score(model_svc, train, y, cv=kfold)
cv_map['SVC'] = scores.mean()

plot = DataFrame.from_dict(data=cv_map, orient='index').plot(kind='bar', legend=False)
pl.show(plot)

# SVC
model_svc.probability = True
model_svc.fit(train_x, train_y)
probas = model_svc.predict_proba(test_x)
fpr, tpr, thresholds = roc_curve(test_y, probas[:, 1])
roc_auc = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC', roc_auc))

# RandomForestClassifier
model_rfc.fit(train_x, train_y)
probas = model_rfc.predict_proba(test_x)
fpr, tpr, thresholds = roc_curve(test_y, probas[:, 1])
roc_auc = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RandonForest', roc_auc))

# KNeighborsClassifier
model_knc.fit(train_x, train_y)
probas = model_knc.predict_proba(test_x)
fpr, tpr, thresholds = roc_curve(test_y, probas[:, 1])
roc_auc = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier', roc_auc))

# LogisticRegression
model_lr.fit(train_x, train_y)
probas = model_lr.predict_proba(test_x)
fpr, tpr, thresholds = roc_curve(test_y, probas[:, 1])
roc_auc = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression', roc_auc))
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc=0, fontsize='small')
pl.show()
