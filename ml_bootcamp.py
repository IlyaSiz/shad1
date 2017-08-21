# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

plt.style.use('ggplot')

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'

# na_values = '?' означает, что данные содержат пропущенные значения, обозначенные символом ?
data = pd.read_csv(url, header=None, na_values='?')

# Узнаем размеры таблицы
print data.shape

# Мы можем посмотреть на несколько первых и несколько последних строк этой таблицы, чтобы получить представление
# об имеющихся данных
print data.head()
print data.tail()

# Согласно описанию рассматриваемой задачи данные содержат информацию о клиентах, запрашивающих кредит.
# Для сохранения конфиденциальности данные обезличены, все значения категориальных признаков заменены символами,
# a числовые признаки приведены к другому масштабу. Последний столбец содержит символы + и -, соответствующие тому,
# вернул клиент кредит или нет.

# Для удобства зададим столбцам имена:
data.columns = ['A' + str(i) for i in range(1, 16)] + ['class']
print data.head()

# К элементам таблицы можно обращаться, например, так:
print data['A5'][687]
print data.at[687, 'A5']

# С помощью метода describe() получим некоторую сводную информацию по всей таблице. По умолчанию будет выдана
# информация только для количественных признаков
print data.describe()

# Заметим, что количество элементов в столбцах A2, A14 меньше общего количества объектов (690), что говорит о том,
# что эти столбцы содержат пропущенные значения.

# Выделим числовые и категориальные признаки:
categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']
print categorical_columns
print numerical_columns

# Теперь мы можем получить некоторую общую информацию по категориальным признакам:
print data[categorical_columns].describe()
# В таблице для каждого категориального признака приведено общее число заполненных ячеек (count), количество значений,
# которые принимает данный признак (unique), самое популярное (часто встречающееся) значение этого признака (top)
# и количество объектов, в которых встречается самое частое значение данного признака (freq).

# Вот немного другой способ получить ту же информацию:
print data.describe(include=[object])

# Определить полный перечень значений категориальных признаков можно, например, так:
for c in categorical_columns:
    print data[c].unique()

# Функция scatter_matrix из модуля pandas.tools.plotting позволяет построить для каждой количественной переменной
# гистограмму, а для каждой пары таких переменных – диаграмму рассеяния:
# plt.show(scatter_matrix(data, alpha=0.05, figsize=(10, 10)).all())

# Из построенных диаграмм видно, что признаки не сильно коррелируют между собой, что впрочем можно также легко
# установить, посмотрев на корреляционную матрицу. Все ее недиагональные значения по модулю не превосходят 0.4:
print data.corr()

# Можно выбрать любую пару признаков и нарисовать диаграмму рассеяния для этой пары признаков, изображая точки,
# соответствующие объектам из разных классов разным цветом: + – красный, - – синий. Например,
# для пары признаков A2, A11 получаем следующую диаграмму:
col1 = 'A2'
col2 = 'A11'
plt.figure(figsize=(10, 6))
plt.scatter(data[col1][data['class'] == '+'],
            data[col2][data['class'] == '+'],
            alpha=0.75,
            color='red',
            label='+')
plt.scatter(data[col1][data['class'] == '-'],
            data[col2][data['class'] == '-'],
            alpha=0.75,
            color='blue',
            label='-')
plt.xlabel(col1)
plt.ylabel(col2)
plt.legend(loc='best')
# plt.show()

# Узнать количество заполненных (непропущенных) элементов можно с помощью метода count. Параметр axis = 0 указывает,
# что мы двигаемся по размерности 0 (сверху вниз), а не размерности 1 (слева направо), т.е. нас интересует количество
# заполненных элементов в каждом столбце, а не строке:
print data.count(axis=0)

# Заполнить пропущенные значения можно с помощью метода fillna
data = data.fillna(data.median(axis=0), axis=0)
print data.count(axis=0)

# Теперь рассмотрим пропущенные значения в столбцах, соответствующих категориальным признакам.
# Простая стратегия – заполнение пропущенных значений самым популярным в столбце. Начнем с A1:
print data['A1'].describe()

# В столбце A1 имеются пропущенные значения. Наиболее частым (встречается 468 раз) является b.
# Заполняем все пропуски этим значением:
data['A1'] = data['A1'].fillna('b')

# Автоматизируем процесс:
data_describe = data.describe(include=[object])
for c in categorical_columns:
    data[c] = data[c].fillna(data_describe[c]['top'])

# Теперь все элементы таблицы заполнены:
print data.describe(include=[object])
print data.describe()

# Как уже отмечалось, библиотека scikit-learn не умеет напрямую обрабатывать категориальные признаки.
# Поэтому прежде чем подавать данные на вход алгоритмов машинного обучения преобразуем категориальные признаки
# в количественные.
# Категориальные признаки, принимающие два значения (т.е. бинарные признаки) и принимающие большее количество значений
# будем обрабатывать по-разному.
# Вначале выделим бинарные и небинарные признаки:
binary_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print binary_columns, nonbinary_columns

# Значения бинарных признаков просто заменим на 0 и 1. Начнем с признака A1:
data.at[data['A1'] == 'b', 'A1'] = 0
data.at[data['A1'] == 'a', 'A1'] = 1
print data['A1'].describe()

# Автоматизируем процесс:
for c in binary_columns[1:]:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1

print data[binary_columns].describe()

# К небинарными признакам применим метод векторизации, который заключается в следующем.
# Признак j, принимающий s значений, заменим на s признаков, принимащих значения 0 или 1, в зависимости от того,
# чему равно значение исходного признака j
# Такую векторизацию осуществляет в pandas метод get_dummies
data_nonbinary = pd.get_dummies(data[nonbinary_columns])
print data_nonbinary.head()

# Нормализация количественных признаков
data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
print data_numerical.describe()

# Соединим все столбцы в одну таблицу:
data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)
print data.shape
print data.columns

# Для удобства отдельно рассмотрим столбцы, соответствующие входным признакам (это будет матрица X),
# а отдельно – выделенный признак (вектор y):
# Выбрасываем столбец 'class'.
X = data.drop('class', axis=1)
y = data['class']
feature_names = X.columns
print feature_names
print X.shape
print y.shape
N, d = X.shape

