#load the data
import datetime
import pandas as pd
import cPickle as pickle
import patsy
import unidecode
import numpy as np
from altair import *

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer


start = datetime.datetime.now()
print '1. loading data'
with open('../99-Reports/006_classification_matrix.p','rb') as f:
    premodel_dict = pickle.load(f)


print '2. assigning variables'
X_train = premodel_dict['X_train']
y_train = premodel_dict['y_train']
X_test = premodel_dict['X_test']
y_test = premodel_dict['y_test']

print '3. initializing the SGD'
sgd = SGDClassifier(loss='log',penalty='l1',n_iter= 100,n_jobs = -1)

print '4. Running SGD --> through one vs. Rest'
model = sgd
OVRC = OneVsRestClassifier(model,n_jobs=-1)
OVRC.fit(X_train,y_train)


print datetime.datetime.now()-start
scores = OVRC.score(X_test,y_test)