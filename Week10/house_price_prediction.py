import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# Linear Regression Refer https://scikitlearn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# for more details. Loading housing dataset in to a dataframe. This dataset has 14 features with 506samples. MEDV is
# the target variable

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

print(df.head())

# Let's use the RM (number of rooms) variable from the Housing dataset as the
# explanatory variable and train a model that can predict MEDV (house prices).
# We only use one feature here, just to explain the concepts and applicability of the model.

X = df[['RM']].values
y = df['MEDV'].values

# Applying Sklearn linear regression model
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


# A function that will plot a scatterplot of the training samples and add the regression line
def lin_reg_plot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.show()


# Plotting MEDV against RM by calling the lin_regplot function
lin_reg_plot(X, y, slr)

# Now we use Random Sample Consensus (RANSAC) algorithm, which identifies
# outliers and fits a regression model to a subset of the data, the so-called inliers.

ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50,
                         loss='absolute_loss', residual_threshold=5.0, random_state=0)
ransac.fit(X, y)
# After we fit the RANSAC model, let's obtain the inliers and outliers from the fitted
# RANSAC-linear regression model and plot them together with the linear fit:
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white',
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white',
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average number of rooms [RM]')

plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()

# Now we use same housing dataset to do the actually implement the model, using training
# and testing sets.
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# Since our model uses multiple explanatory variables, we can't visualize the linear
# regression line (or hyperplane to be precise) in a two-dimensional plot, but we can plot
# the residuals (the differences or vertical distances between the actual and predicted
# values) versus the predicted values to diagnose our regression model.

plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o',
            edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s',
            edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

# Using MSE(Mean Square Error to measure the performance)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
############################################################

# You can refer https://scikitlearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html to
# understand on Logistic Regression
