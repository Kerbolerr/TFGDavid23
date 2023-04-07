import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import *
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn import neighbors
 
import warnings,joblib
from fastapi import FastAPI

warnings.filterwarnings('ignore')

df = pd.read_excel('./CleanData/Test/DesktopTest.ods',engine="odf")

print(df.head)

columns = list(df)
features = df[columns[0:len(columns)-1]]
target = df[[columns[len(columns)-1]]]
 
scaler = preprocessing.StandardScaler()
scaler2 = preprocessing.StandardScaler()

features = scaler.fit_transform(features)
target = scaler2.fit_transform(target)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.2, random_state=2022)
print(X_train.shape, X_valid.shape)

print(X_train[0])

models = [LinearRegression]
for i in range(0,len(models)):
  clf = ensemble.RandomForestRegressor()
  clf = clf.fit(X_train, Y_train)
#print(sum(abs(list(Y_valid)-clf.predict(X_valid))))
print(clf.score(X_valid,Y_valid))


joblib.dump(scaler,"scalerFeaturesDesktop.skl")
joblib.dump(scaler2,"scalerTargetDesktop.skl")
joblib.dump(clf,"Desktop.sav")

