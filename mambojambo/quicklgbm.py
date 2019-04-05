import pandas as pd
import numpy as np
import TrainLgbmBinaryClassifier

### this is a small extract from the ember dataset
X_train = pd.read_csv("datasets/X_train.csv.gzip",compression="gzip")
y_train = pd.read_csv("datasets/y_train.csv.gzip",compression="gzip")
X_test = pd.read_csv("datasets/X_test.csv.gzip",compression="gzip")
y_test = pd.read_csv("datasets/y_test.csv.gzip",compression="gzip")

data={}
data['X_train']=X_train
data['X_test']=X_test
data['y_train']=y_train.iloc[:,1]
data['y_test']=y_test.iloc[:,1]
ret = TrainLgbmBinaryClassifier.train(data,folds=2)
