from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------

"""
Our example dataset.
We've already made the lag 1 for you.
"""

"""
again, in this example
"""

df = pd.read_csv("./output/cleaned_best_feature_data.csv")
df = df[["Relative humidity (%)",
         'Wind maximum (m/s)',
         'NO (ug/m3)',
         'NO2 (ug/m3)',
         'Temperature 2m (DegC)',
         "PM10 (ug/m3)"]]
"""
from sklearn.model_selection import TimeSeriesSplit

We may also want to use TimeSeriesSplit 

ts_cv = TimeSeriesSplit(
    n_splits=2,
    gap=48,
    max_train_size=10000,
    test_size=1000,
)

It ensures that chopping the data into windows of consecutive samples is still possible.

It ensures that the validation/test results are more realistic, 
being evaluated on the data collected after the model was trained.

in this example, we just chop it off ( training: 70%,  30% )
"""
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n * 0.7)]  # 0 ~~ 70%
test_df = df[int(n * 0.7):]  # 70% ~~ end

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
# or boxplot
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.show()
plt.savefig('./output/st_albans_violinplot.png', dpi=800)

# I randomly pick this 5 features,
# it does not mean you will use it.
# please use your own feature selection algorithms

feature_cols = ["Relative humidity (%)",
                'Wind maximum (m/s)',
                'NO (ug/m3)',
                'NO2 (ug/m3)',
                'Temperature 2m (DegC)']

value_col = ['PM10 (ug/m3)']

train_features = train_df[feature_cols].values
train_y = train_df[value_col].values

test_features = test_df[feature_cols].values
test_y = test_df[value_col].values

# let's say : first layer : 100 neron, second layer : 50 neron, third layer : 25 neron
model = MLPRegressor(hidden_layer_sizes=(100, 50, 25))
model.fit(train_features, train_y)

# make predictions and find the rmse
preds = model.predict(test_features)
rms = np.sqrt(np.mean(np.power((np.array(test_y) - np.array(preds)), 2)))
print("root meam squre error", rms)

MSE = np.square(np.subtract(test_y, preds)).mean()
print("Mean Squared Error", MSE)

correlation_matrix = np.corrcoef(
    np.reshape(test_y, test_y.shape[0]),
    np.reshape(preds, preds.shape[0])
)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy ** 2
print('RSquare:', r_squared)

"""
This is a very simple model


Please complete your assignment with a better one

refer to this tutorial
 
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlpregressor

"""
