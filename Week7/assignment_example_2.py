import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import classification_report, confusion_matrix
from scipy.io import arff
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

"""
    Assignment 2 
     Part B
"""

"""
    (1)
    from scipy.io import arff
    
    convert arff data
    data = arff.loadarff('iris.arff')  
"""

"""
    (a) autism spectrum disorder (ASD)
    literature review combines with your data visualization
    Seem as above !
"""

"""
    (b) feature selections, same as above example (try different feature selection algorithms)
"""

"""
    (c) GaussianNB
"""
iris = arff.loadarff('iris.arff')
iris = pd.DataFrame(iris[0])

print(iris.shape)
print(iris.head(3))

le = LabelEncoder()
le.fit(iris['class'])
iris['class'] = le.transform(iris['class'])

trainData, testData, trainTarget, testTarget = train_test_split(iris.values[:, 0:3],
                                                                iris['class'].values,
                                                                test_size=1 / 3)

classifier = GaussianNB()
classifier.fit(trainData, trainTarget)

predictedValues = classifier.predict(testData)

# print the classification_report ,accuracy
print(classification_report(testTarget, predictedValues))
print(confusion_matrix(testTarget, predictedValues))

"""
     precision    recall  f1-score   support
           0       1.00      1.00      1.00        20
           1       0.72      0.93      0.81        14
           2       0.92      0.69      0.79        16
    accuracy                           0.88        50
   macro avg       0.88      0.87      0.87        50
weighted avg       0.90      0.88      0.88        50

        [
         [20  0  0]
         [ 0 13  1]
         [ 0  5 11]
        ]
        
"""

"""
    (d)  feature selections  (in part (c))
          VS 
         tree. DecisionTreeClassifier.feature_importances_ (same as above)
"""
