import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score,auc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,confusion_matrix

path="Week3/Lab3/breast-cancer.xlsx"
rawdata= pd.read_excel(path)
#print(rawdata)
#print ("data summary")
#print (rawdata.describe())
nrow, ncol = rawdata.shape
#print (nrow, ncol)


class_le = LabelEncoder()
y = class_le.fit_transform(rawdata['class'].values)
#print(y)

X= rawdata[['age', 'menopause', 'tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']].values
ohe = ColumnTransformer([('anyname', OneHotEncoder(), [0,1,2,3,4,5,6,7,8])], remainder = 'passthrough')
#print (ohe.fit_transform(X))
newdata = ohe.fit_transform(X)
#print(newdata)

predictors = newdata
target = y
pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target, test_size=.3)
#print(predictors)
#print(target)

# Decision Tree
classifier = DecisionTreeClassifier(criterion="entropy", random_state=999, min_samples_split=4)
classifier = classifier.fit(pred_train, tar_train)
predictions = classifier.predict(pred_test)
prob = classifier.predict_proba(pred_test)
tn, fp, fn, tp = confusion_matrix(tar_test,predictions).ravel()
print(tn, fp, fn, tp)

print("Accuracy score of our model with Decision Tree:", accuracy_score(tar_test, predictions))
for x in range(2):
    precision = precision_score(y_true=tar_test, y_pred=predictions,average='binary', pos_label=x)
    print("Precision score for class", x, "with Decision Tree :", precision)
    recall = recall_score(y_true=tar_test, y_pred=predictions,average='binary', pos_label=x)
    print("Recall score for class", x, " with Decision Tree :", recall)



