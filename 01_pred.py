# -*- coding: utf-8 -*-

# Предобработка данных в Pandas

import numpy as np
import pandas


# загрузка данных в Pandas:
data = pandas.read_csv('pred/titanic.csv', index_col='PassengerId')

# 1. Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
# 577 314
print data['Sex'].value_counts()

# 2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
# Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой, например, 0.42.
# При необходимости округляйте дробную часть до двух знаков.
# 38.38
survive = data['Survived'].value_counts()
surCounts = survive[1]
# Всего 891 пассажир
allCounts = sum(survive)
print surCounts.astype(np.float64) / allCounts.astype(np.float64) * 100

# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
# 24.24
classes = data['Pclass'].value_counts()
firstClassCounts = classes[1]
print firstClassCounts.astype(np.float64) / allCounts.astype(np.float64) * 100

# 4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.
# 29.7 28.0
ages = data['Age']
# ages = ages[ages.notnull()]
age_mean = ages.mean()
age_median = ages.median()
print age_mean, age_median

# 5. Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
# 0.41
c = data[['SibSp', 'Parch']]
print c.corr()

# 6. Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name)
# его личное имя (First Name). Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
# Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию.
# Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен,
# а также разделения их на женские и мужские.
# Anna 15
female_names = data[data['Sex'] == 'female']['Name']
full_names = female_names.str.split(',', expand=True)[1]
names = full_names.str.split('.', expand=True)[1]
persNames = names.str.split('(', expand=True)

persNames = persNames[1].combine_first(persNames[0]).str.strip().str.split(' ', expand=True)
n = persNames[0].str.strip(')')
print n.value_counts()
