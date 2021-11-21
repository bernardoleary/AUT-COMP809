from scipy.io import arff
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns

"""
Assessment 1 part 2,

Q: how to load arff file?  
A: scipy.io  to load the file, then convert it to dataframe

"""

data = arff.loadarff('Week5/Lab5/iris.arff')
df = pd.DataFrame(data[0])

print(df.head())

"""
Assessment 1 part 2,

Q: what are the most influential features
A: features selection algorithms (we only show you 2 examples, but you should do more)

"""

data = pd.read_csv("./train.csv")
X = data.iloc[:, 0:20]  # independent columns
y = data.iloc[:, -1]  # target column i.e price range
# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(10, 'Score'))  # print 10 best features

# ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers

# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

"""
Assessment 1 part 2,

Q: get correlations of each features in dataset
A: plot the correlations matrix and Interpret it.

"""

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
# plot heat map
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()
