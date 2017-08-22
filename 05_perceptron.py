# -*- coding: utf-8 -*-
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# загрузка данных в Pandas:
data_train = pandas.read_csv('perceptron/perceptron-train.csv', header=None)
data_test = pandas.read_csv('perceptron/perceptron-test.csv', header=None)

# Pandas Dataframe to np.array
X_train = data_train.ix[:, 1:].values
y_train = data_train.ix[:, :0].values.ravel()
X_test = data_test.ix[:, 1:].values
y_test = data_test.ix[:, :0].values.ravel()

# Создаем и обучаем персептрон
clf = Perceptron(random_state=241)
clf.fit(X=X_train, y=y_train)
predictions_test = clf.predict(X_test)
acc_before_scale = accuracy_score(y_true=y_test, y_pred=predictions_test)

# Нормализуем выборку
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf.fit(X=X_train_scaled, y=y_train)
predictions_test_scaled = clf.predict(X_test_scaled)
acc_after_scale = accuracy_score(y_true=y_test, y_pred=predictions_test_scaled)

print acc_before_scale, acc_after_scale
result = acc_after_scale - acc_before_scale
print result
