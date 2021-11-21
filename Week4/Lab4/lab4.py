import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, auc 
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.datasets import load_iris

path = "Lab4/Iris.xlsx" #should change the pathaccordingly
rawdata = pd.read_excel(path) #pip install xlrd
iris = load_iris()
X, y = iris.data, iris.target
pred_train, pred_test, tar_train, tar_test = train_test_split(X, y, test_size=.3)

 #Create a Random Forest Classifier 
 # https://www.datacamp.com/community/tutorials/random-forests-classifier-python
clf = RandomForestClassifier(n_estimators=100)

 #Train the model using the training sets 
clf.fit(pred_train, tar_train) 

 # prediction on test set
predictions = clf.predict(pred_test) 

 # Model Accuracy 
print("Accuracy score of our model with Random Forest:", accuracy_score(tar_test, predictions))

gnb = GaussianNB() #suitable for numeric features
gnb.fit(pred_train, np.ravel(tar_train,order='C'))
predictions = gnb.predict(pred_test)
print("Accuracy score of our model with Gaussian Naive Bayes:", accuracy_score(tar_test, predictions))

nbrs = KNeighborsClassifier()
nbrs.fit(pred_train,np.ravel(tar_train,order='C'))
predictions = nbrs.predict(pred_test)
print("Accuracy score of our model with kNN :", accuracy_score(tar_test, predictions))

clf = MLPClassifier()
clf.fit(pred_train,np.ravel(tar_train,order='C'))
predictions = clf.predict(pred_test)
print("Accuracy score of our model with MLP :", accuracy_score(tar_test, predictions))
scores = cross_val_score(clf, X, y, cv=10)
print("Accuracy score of our model with MLP under cross validation :", scores.mean())