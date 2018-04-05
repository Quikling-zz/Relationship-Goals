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

filenameX = 'X.csv'
filenamey = 'y.csv'

data = load_data(filenameX, filenamey, header=1)

X, y = data.X, data.y

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

svc_train_score = 0.0
svc_test_score = 0.0
dummy_train_score = 0.0
dummy_test_score = 0.0

for train, test in kf.split(X, y):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    dumclf = DummyClassifier(strategy="most_frequent")
    dumclf.fit(X_train, y_train)

    clf = SVC()
    clf.fit(X_train, y_train)

    svc_train_score += clf.score(X_train, y_train)
    svc_test_score += clf.score(X_test, y_test)
    dummy_train_score += dumclf.score(X_train, y_train)
    dummy_test_score += dumclf.score(X_test, y_test)


print "SVC train: %.6f" % (svc_train_score/n_splits)
print "Dummy train: %.6f" % (dummy_train_score/n_splits)    
print "SVC test: %.6f" % (svc_test_score/n_splits)
print "Dummy test: %.6f" % (dummy_test_score/n_splits)
