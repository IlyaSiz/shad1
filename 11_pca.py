# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA
from numpy import corrcoef

# Введение

# Метод главных компонент (principal component analysis, PCA) — это один из методов обучения без учителя,
# который позволяет сформировать новые признаки, являющиеся линейными комбинациями старых.
# При этом новые признаки строятся так, чтобы сохранить как можно больше дисперсии в данных.
# Иными словами, метод главных компонент понижает размерность данных оптимальным
# с точки зрения сохранения дисперсии способом.

# Основным параметром метода главных компонент является количество новых признаков.
# Как и в большинстве методов машинного обучения, нет четких рекомендаций по поводу выбора значения этого параметров.
# Один из подходов — выбирать минимальное число компонент, при котором объясняется не менее определенной доли дисперсии
# (это означает, что в выборке сохраняется данная доля от исходной дисперсии).

# В этом задании понадобится измерять схожесть двух наборов величин. Если имеется набор пар измерений
# (например, одна пара — предсказания двух классификаторов для одного и того же объекта),
# то охарактеризовать их зависимость друг от друга можно с помощью корреляции Пирсона.
# Она принимает значения от -1 до 1 и показывает, насколько данные величины линейно зависимы.
# Если корреляция равна -1 или 1, то величины линейно выражаются друг через друга.
# Если она равна нулю, то линейная зависимость между величинами отсутствует.

# Данные

# В этом задании мы будем работать с данными о стоимостях акций 30 крупнейших компаний США.
# На основе этих данных можно оценить состояние экономики, например, с помощью индекса Доу-Джонса.
# Со временем состав компаний, по которым строится индекс, меняется. Для набора данных был взят период
# с 23.09.2013 по 18.03.2015, в котором набор компаний был фиксирован
# (подробнее почитать о составе можно по ссылке из материалов).

# Одним из существенных недостатков индекса Доу-Джонса является способ его вычисления — при подсчёте индекса
# цены входящих в него акций складываются, а потом делятся на поправочный коэффициент. В результате, даже если одна
# компания заметно меньше по капитализации, чем другая, но стоимость одной её акции выше, то она сильнее влияет
# на индекс. Даже большое процентное изменение цены относительно дешёвой акции может быть нивелировано
# незначительным в процентном отношении изменением цены более дорогой акции.

# Реализация в sklearn

# Метод главных компонент реализован в пакете scikit-learn в модуле decomposition в классе PCA.
# Основным параметром является количество компонент (n_components). Для обученного преобразования этот класс
# позволяет вычислять различные характеристики. Например, поле explained_variance_ratio_ содержит процент дисперсии,
# который объясняет каждая компонента. Поле components_ содержит информацию о том, какой вклад вносят признаки
# в компоненты. Чтобы применить обученное преобразование к данным, можно воспользоваться методом transform.

# Для нахождения коэффициента корреляции Пирсона можно воспользоваться функцией corrcoef из пакета numpy.

# Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний на закрытии торгов за каждый
# день периода.
df = pd.read_csv('pca/close_prices.csv')
X = df.drop('date', axis=1)

# На загруженных данных обучите преобразование PCA с числом компоненты равным 10.
model = PCA(n_components=10)
model.fit(X)
print model.explained_variance_ratio_

# Скольких компонент хватит, чтобы объяснить 90% дисперсии?
var = 0.0
n_var = 0
for v in model.explained_variance_ratio_:
    n_var += 1
    var += v
    if var >= 0.9:
        break
print(n_var)

# Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.
X_transformed = pd.DataFrame(model.transform(X))
# print X_transformed

# Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv.
df2 = pd.read_csv('pca/djia_index.csv')
dow_jones_ind = df2['^DJI']

# Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
corr = corrcoef(dow_jones_ind, X_transformed[0])
print '{:0.2f}'.format(corr[0, 1])

# Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.
comp_index = pd.Series(model.components_[0]).sort_values(ascending=False).head(1).index[0]
company = X.columns[comp_index]
print company
