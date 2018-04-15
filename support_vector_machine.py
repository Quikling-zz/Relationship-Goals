# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle
from utils import *

from collections import defaultdict

filenameX = 'X.csv'
filenamey = 'y.csv'

data = load_data(filenameX, filenamey, header=1)

X, y = data.X, data.y

class_count = defaultdict(float)
for example in y:
    class_count[example] += 1.0

weights = []

for example in y:
    weights.append(class_count[example])

divideWeight = class_count[max(class_count)]

data.weights = np.array(map(lambda x: divideWeight/x, weights))


n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

svc_train_score = 0.0
svc_test_score = 0.0
dummy_train_score = 0.0
dummy_test_score = 0.0

for train, test in kf.split(X, y):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    train_weights, test_weights = data.weights[train], data.weights[test]



    dumclf = DummyClassifier(strategy="most_frequent")
    dumclf.fit(X_train, y_train, sample_weight=train_weights)

    clf = SVC(class_weight="balanced")
    clf.fit(X_train, y_train)
    y_label = clf.predict(X_test)

    svc_train_score += clf.score(X_train, y_train)
    svc_test_score += clf.score(X_test, y_test)
    dummy_train_score += dumclf.score(X_train, y_train, train_weights)
    dummy_test_score += dumclf.score(X_test, y_test, test_weights)

    # compute classifier performance
    metricss = ['f1_score', 'precision', 'sensitivity', 'specificity']
    for metric in metricss:
        if metric=="f1_score":
            score = metrics.f1_score(y_test, y_label, average='binary')
        if metric=='precision':
            score = metrics.precision_score(y_test, y_label, average='binary')
        if metric=='sensitivity':
            M = metrics.confusion_matrix(y_test, y_label, labels=[1, 0])
            score = M[0][0]*1.0/sum(M[0])
        if metric=='specificity':
            M = metrics.confusion_matrix(y_test, y_label, labels=[1, 0])
            score = M[1][1]*1.0/sum(M[1])
        print metric, score


print "SVC train: %.6f" % (svc_train_score/n_splits)
print "Dummy train: %.6f" % (dummy_train_score/n_splits)
print "SVC test: %.6f" % (svc_test_score/n_splits)
print "Dummy test: %.6f" % (dummy_test_score/n_splits)
