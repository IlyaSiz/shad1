# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import pandas as pd

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
data = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
feature_names = vectorizer.get_feature_names()

# df = pandas.DataFrame(denselist, columns=feature_names, index=characters)

# idf = vectorizer.idf_
# print len(idf)
# print dict(zip(vectorizer.get_feature_names(), idf))

# cv = KFold(n_splits=5, shuffle=True, random_state=241)
# clf = SVC(kernel='linear', random_state=241)
# grid = {'C': np.power(10.0, np.arange(-5, 6))}
# gs = GridSearchCV(estimator=clf, param_grid=grid, scoring='accuracy', cv=cv)
# gs.fit(X, y)

# ['rank_test_score', 'split4_test_score', 'mean_score_time', 'std_test_score', 'std_train_score',
# 'split1_train_score', 'split0_test_score', 'mean_test_score', 'std_score_time', 'split2_train_score',
# 'param_C', 'split0_train_score', 'params', 'std_fit_time', 'split4_train_score', 'split2_test_score',
# 'split3_test_score', 'mean_train_score', 'mean_fit_time', 'split3_train_score', 'split1_test_score']

# cv_results = gs.cv_results_
# test_scores = cv_results.get('mean_test_score')
# params_c = cv_results.get('param_C')
# print test_scores
# print params_c
# [ 0.55263158  0.55263158  0.55263158  0.55263158  0.95016797  0.99328108  0.99328108  0.99328108  0.99328108
#   0.99328108  0.99328108]
# [1.0000000000000001e-05 0.0001 0.001 0.01 0.10000000000000001 1.0 10.0 100.0 1000.0 10000.0 100000.0]
print '***************************************************'
# max_score = max(test_scores)
# max_ind = test_scores.index(max_score)
# print max_score, params_c[max_ind]
# print '***************************************************'
# print 'CV error    = ', 1 - gs.best_score_
# print 'best C      = ', gs.best_estimator_.C
# CV error    =  0.006718924972
# best C      =  1.0

# clf = SVC(kernel='linear', C=gs.best_estimator_.C, random_state=241)
clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(X, y)

top10idx = np.array(clf.coef_.indices)[np.abs(np.array(clf.coef_.data)).argsort()[-10:]]
print top10idx
words = [feature_names[x] for x in top10idx]
words.sort()
print(words)
# atheism atheists bible god keith moon nick religion sky space
