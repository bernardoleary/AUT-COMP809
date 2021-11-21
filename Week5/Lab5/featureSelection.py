

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


# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
rawdata = pandas.read_csv(url, names=names)
array = rawdata.values
nrow, ncol = rawdata.shape
X = array[:, 0:8]
Y = array[:, 8]

# generate model and get accuracy

def get_accuracy(target_train, target_test, predicted_test,predicted_train):
    clf = MLPClassifier(activation='logistic', solver='sgd', learning_rate_init=0.1, alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1,max_iter=2000)
    clf.fit(predicted_train, np.ravel(target_train, order='C'))
    predictions = clf.predict(predicted_test)
    return accuracy_score(target_test, predictions)


pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y, test_size=.3, random_state=4)
print("Accuracy score of our model without feature selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,
                                                                                    pred_train))

# feature extraction
test = SelectKBest(score_func=chi2, k=3)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:3, :],"summerize features")
print()

# Now apply only the K most significant features according to the chi square method
pred_features = features[:, 0:3]
pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
print("Accuracy score of our model with chi square feature selection : %.2f" % get_accuracy(tar_train, tar_test,
                                                                                            pred_test,pred_train))
print()
from sklearn.feature_selection import SelectFromModel
# Feature Extraction with RFE
model = LogisticRegression()  # Logistic regression is the Wrapper classifier here
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
#print("Num Features: %d" % (fit.n_features_))
#print("Selected Features: %s" % (fit.support_))
#print("Feature Ranking: %s" % (fit.ranking_))
##Now apply only the K most significant features according to the RFE feature selection method
features = fit.transform(X)
pred_features = features[:, 0:3]
#
pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
print("Accuracy score of our model with RFE selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,pred_train))
print()

## Feature Extraction with PCA


## feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
features = fit.transform(X)
## summarize components
#print("Explained Variance: %s" % (fit.explained_variance_ratio_))
#print(fit.components_)
#
##Now apply only the K most significant faetures (components) according to the PCA feature selection method
#features = fit.transform(X)
pred_features = features[:, 0:3]
pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
print("Accuracy score of our model with PCA selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,pred_train))
print()
#
## Feature Importance with Extra Trees Classifier

from sklearn.ensemble import ExtraTreesClassifier

## feature extraction
model = ExtraTreesClassifier(max_depth=3,min_samples_leaf=2)
fit = model.fit(X, Y)
print(model.feature_importances_)
print()
t = SelectFromModel(fit, prefit=True)
features = t.transform(X)
pred_features = features[:, 0:3]

pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
print("Accuracy score of our model with Extra Trees selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,
                                                                                     pred_train))
#print()