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


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

for train, test in kf.split(X, y):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    dumclf = DummyClassifier(strategy="most_frequent")
    dumclf.fit(X_train, y_train)

    clf = SVC()
    clf.fit(X_train, y_train)

    print "SVC: %.6f" % (clf.score(X_test, y_test))
    print "Dummy: %.6f" % (dumclf.score(X_test, y_test))
