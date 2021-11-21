import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import precision_score, recall_score, auc
from sklearn.metrics import roc_curve, accuracy_score
import seaborn as sns

# from sklearn.datasets import load_iris
# import numpy as np
# from sklearn import tree


path = "Week3/Lab3/Iris.xlsx"  # should change the path accordingly
rawdata = pd.read_excel(path)  # pip install xlrd
print("data summary")
print(rawdata.describe())
nrow, ncol = rawdata.shape
print(nrow, ncol)

# save load_iris() sklearn dataset to iris
# if you'd like to check dataset type use: type(load_iris())
# if you'd like to view list of attributes use: dir(load_iris())
# iris = load_iris()

# np.c_ is the numpy concatenate function
# which is used to concat iris['data'] and iris['target'] arrays
# for pandas column argument: concat iris['feature_names'] list
# and string list (in this case one string); you can make this anything you'd like..
# the original dataset would probably call this ['Species']
# rawdata = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

print("\n correlation Matrix")
corr = rawdata.corr()
print(corr)

"""
-1 indicates a perfectly negative linear correlation between two variables.
0  indicates no linear correlation between two variables.
1  indicates a perfectly positive linear correlation between two variables.

-----

Interpretation
Diagonal values and upper triangle are ignored (melted the upper triangle through np.tril and df.where).
Naturally, we find:

a high positive correlation between PetalWidth and PetalLength (0.96)
a high positive correlation between PetalLength and SepalLength (0.87)
a high positive correlation between PetalWidth and SepalLength (0.81)
As such, we observe correlations between these main attributes: PetalWidth, PetalLength and SepalLength.

"""

rawdata.hist(stacked=True, bins=10)
plt.subplots_adjust(hspace=0.5)
plt.show()

sns.FacetGrid(rawdata, hue="class", height=3).map(sns.distplot, "sepal_len").add_legend()
plt.show()
sns.FacetGrid(rawdata, hue="class", height=3).map(sns.distplot, "sepal_wid").add_legend()
plt.show()
sns.FacetGrid(rawdata, hue="class", height=3).map(sns.distplot, "petal_len").add_legend()
plt.show()
sns.FacetGrid(rawdata, hue="class", height=3).map(sns.distplot, "petal_wid").add_legend()
plt.show()

"""
https://medium.com/analytics-vidhya/exploratory-data-analysis-uni-variate-analysis-of-iris-data-set-690c87a5cd40
"""

"""
matplotlib example
https://matplotlib.org/

"""
pd.plotting.scatter_matrix(rawdata, figsize=[8, 8])
plt.show()

"""
seaborn example
https://seaborn.pydata.org/
"""
sns.set_style("whitegrid")
sns.pairplot(rawdata, hue="class", height=3, diag_kind="hist", markers=["o", "s", "D"], palette="Set2")
plt.show()

# boxplot
fig = plt.figure(1, figsize=(12, 10))
ax = fig.add_subplot(111)
ax.boxplot(rawdata.values)
ax.set_xticklabels(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class'])
plt.show()

predictors = rawdata.iloc[:, :ncol - 1]
print(predictors)
target = rawdata.iloc[:, -1]
print(target)

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, stratify=target, test_size=.3)

# Decision Tree
split_threshold = 4
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2, split_threshold):
    classifier = DecisionTreeClassifier(criterion="entropy", random_state=999, min_samples_split=i)
    classifier = classifier.fit(pred_train, tar_train)
    predictions = classifier.predict(pred_test)
    prob = classifier.predict_proba(pred_test)
    text = export_text(classifier, feature_names=None, max_depth=10, spacing=3, decimals=2, show_weights=False)
    print(text)
    """
      instead of
       large amount of hard-coded
            if-else code
      Decision Tree can learn the logic from the raw data

      It is highly interpretable. However, the interpretability decreases as the depth of the tree increases
    """
    # tree.plot_tree(classifier)
    print(prob, i, "min sample")

    print("Accuracy score of our model with Decision Tree:", i, accuracy_score(tar_test, predictions))
    precision = precision_score(y_true=tar_test, y_pred=predictions, average='micro')
    print("Precision score of our model with Decision Tree :", precision)
    recall = recall_score(y_true=tar_test, y_pred=predictions, average='micro')
    print("Recall score of our model with Decision Tree :", recall)

    """
    class 0
    class 1
    class 2
    """

    for x, class_name in enumerate(["Iris-setosa", "Iris-versicolor", "Iris-virginica"]):
        # To draw the ROC curve, one of the response column class has to be the positive one.
        # The parameter 'pos_label' represents this class.
        fpr[x], tpr[x], _ = roc_curve(tar_test[:], prob[:, x], pos_label=x)
        # print(roc_curve(tar_test[:], prob[:, x],pos_label=x))
        roc_auc[x] = auc(fpr[x], tpr[x])
        print("AUC values of the decision tree ", class_name, " : ", roc_auc[x])

        plt.plot(fpr[x], tpr[x], color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc[x])
        plt.show()
