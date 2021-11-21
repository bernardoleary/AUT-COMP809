
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# =================================================
# Data pre-processing
# Apply semi-colon delimitaion, remove the records that were classified "unknown" 
# Turn all parameters into categorical (numerical) values
# =================================================
# load data
path = "Ass1/PartB/bank/bank.csv"
rawdata = pandas.read_csv(path, sep=';')

# filter to exclude records where the outcome of the marketing campaign was unknown 
known_outcomes = rawdata[rawdata["poutcome"] != "unknown"]

list_of_columns = known_outcomes.columns
known_outcomes[list_of_columns] = known_outcomes[list_of_columns].apply(lambda col:pandas.Categorical(col).codes)

array = known_outcomes.values
nrow, ncol = known_outcomes.shape
X = array[:, 0:16]
Y = array[:, 16]

# =================================================
# Feature selection
# Feature Extraction with RFE with LogisticRegression Wrapper gives the best accuracy of 0.8
# =================================================

# generate model and get accuracy
def get_accuracy(target_train, target_test, predicted_test,predicted_train):
    clf = MLPClassifier(activation='logistic', solver='sgd', learning_rate_init=0.1, alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1,max_iter=2000)
    clf.fit(predicted_train, np.ravel(target_train, order='C'))
    predictions = clf.predict(predicted_test)
    return accuracy_score(target_test, predictions)

pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y, test_size=.3, random_state=4)
print("Accuracy score of our model without feature selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test, pred_train))

# feature extraction
test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5, :],"summerize features")
print()
# Now apply only the K most significant features according to the chi square method
pred_features = features[:, 0:5]
pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
print("Accuracy score of our model with chi square feature selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,pred_train))
print()

## Feature Importance with Extra Trees Classifier
from sklearn.feature_selection import SelectFromModel
# Feature Extraction with RFE
model = LogisticRegression()  # Logistic regression is the Wrapper classifier here
rfe = RFE(model, 5)
fit = rfe.fit(X, Y)
## summarize components
#print("Num Features: %d" % (fit.n_features_))
#print("Selected Features: %s" % (fit.support_))
#print("Feature Ranking: %s" % (fit.ranking_))
## Now apply only the K most significant features according to the RFE feature selection method
features = fit.transform(X)
pred_features = features[:, 0:5]
pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
print("Accuracy score of our model with RFE selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,pred_train))
print()

## Feature Extraction with PCA
## feature extraction
pca = PCA(n_components=5)
fit = pca.fit(X)
features = fit.transform(X)
## summarize components
#print("Explained Variance: %s" % (fit.explained_variance_ratio_))
#print(fit.components_)
## Now apply only the K most significant faetures (components) according to the PCA feature selection method
#features = fit.transform(X)
pred_features = features[:, 0:5]
pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
print("Accuracy score of our model with PCA selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,pred_train))
print()

## Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
## feature extraction
model = ExtraTreesClassifier(max_depth=3,min_samples_leaf=2)
fit = model.fit(X, Y)
print(model.feature_importances_)
print()
t = SelectFromModel(fit, prefit=True)
features = t.transform(X)
pred_features = features[:, 0:5]
pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
print("Accuracy score of our model with Extra Trees selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test, pred_train))
print()


# =================================================
# DTC
# Parameters adjusted are 
# =================================================

## Feature Importance with Extra Trees Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, auc
from sklearn.metrics import accuracy_score,confusion_matrix

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