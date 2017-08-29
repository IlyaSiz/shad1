# -*- coding: utf-8 -*-
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

# 1. Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv.
train = pandas.read_csv('linreg/salary-train.csv')

# 2. Проведите предобработку:


def text_transform(text):
    # Приведите тексты к нижнему регистру (text.lower()).
    text = text.map(lambda t: t.lower())

    # Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова. Для такой
    # замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text). Также можно воспользоваться
    # методом replace у DataFrame, чтобы сразу преобразовать все тексты:
    text = text.replace('[^a-zA-Z0-9]', ' ', regex=True)
    return text

# Примените TfidfVectorizer для преобразования текстов в векторы признаков. Оставьте только те слова,
# которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).

vectorizer = TfidfVectorizer(min_df=5)
X_train_text = vectorizer.fit_transform(text_transform(train['FullDescription']))

# Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'.

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

# Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
oneHotDecoder = DictVectorizer()
X_train_cat = oneHotDecoder.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Объедините все полученные признаки в одну матрицу "объекты-признаки". Обратите внимание, что матрицы для текстов и
# категориальных признаков являются разреженными. Для объединения их столбцов нужно воспользоваться функцией
# scipy.sparse.hstack.
X_train = hstack([X_train_text, X_train_cat])

# 3. Обучите гребневую регрессию с параметром alpha=1. Целевая переменная записана в столбце SalaryNormalized.
y_train = train['SalaryNormalized']
model = Ridge(alpha=1)
model.fit(X_train, y_train)

# 4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv. Значения полученных прогнозов являются
# ответом на задание. Укажите их через пробел.

test = pandas.read_csv('linreg/salary-test-mini.csv')
X_test_text = vectorizer.transform(text_transform(test['FullDescription']))
X_test_cat = oneHotDecoder.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)
print('{:0.2f} {:0.2f}'.format(y_test[0], y_test[1]))
# 56589.01 37186.84
