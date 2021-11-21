import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("./output/cleaned_best_feature_data.csv")

# creating a separate dataset
new_data = data[['DateTime', 'PM10 (ug/m3)']]
# reset index to the timestamp, instead of numeric sequential values like : 1,2,3,4
new_data.index = new_data.DateTime
new_data.drop('DateTime', axis=1, inplace=True)

# creating train and test sets
dataset = new_data.values

train = dataset[0:3500, :]
valid = dataset[3500:, :]

# It’s a good idea to normalize the data before model fitting.
# This will boost the performance. You can read more here for the Min-Max Scaler:
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

"""
https://www.tensorflow.org/tutorials/structured_data/time_series

Now the time-step value will be 60. Let’s split the data X, Y.
In the 0th iteration the 
first 60 elements goes as your first record
 and the 61 elements will be put up in the prediction. 

 1 to 60 as a batch to fit the closing price on timestamp 61
 2 to 61 as a batch to fit the closing price on timestamp 62
 3 to 62 as a batch to fit the closing price on timestamp 63
 4 to 63 as a batch to fit the closing price on timestamp 64
 ........ 
"""
time_step = 50  # step = 100 ? 600 ? whatever

# converting dataset into x_train and y_train
x_train, y_train = [], []
for i in range(time_step, len(train)):
    x_train.append(scaled_data[i - time_step:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# create and fit the very simple LSTM network
# Now, it’s time to build the model.
# We will build the LSTM with 50 neurons and 2 hidden layers.
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

"""
In next assignment,
for all the models, provide root mean square error (RMSE),

Mean Absolute Error (MAE) and correlation
coefficient (R2 ) to quantify the prediction performance of each model.

mean_absolute_error, RootMeanSquaredError

from tensorflow_addons.metrics import RSquare
https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/RSquare
"""
model.compile(loss='mean_squared_error',
              optimizer=Adam(learning_rate=0.01),
              # metrics=MeanAbsoluteError()  # r_square(), mean_absolute_error(), RootMeanSquaredError()
              )

model.fit(x_train,
          y_train,
          epochs=1,  # 100
          batch_size=4
          )

# Prepare the test data (reshape them):
# predicting values, using past time_step from the train data
inputs = new_data[len(new_data) - len(valid) - time_step:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(time_step, inputs.shape[0]):
    X_test.append(inputs[i - time_step:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pm10 = model.predict(X_test)
# because we have scaled the data, so we should inverse the scale
pm10 = scaler.inverse_transform(pm10)

# we can get the error, in this example, it is mean_absolute_error:
# print('Mean Absolute Error:', model.evaluate(X_test))

# you can also calculate the error by formula, such as rmse
rms = np.sqrt(np.mean(np.power((valid - pm10), 2)))
print('rmse:', rms)

correlation_matrix = np.corrcoef(
    np.reshape(valid, valid.shape[0]),
    np.reshape(pm10, pm10.shape[0])
)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy ** 2
print('RSquare:', r_squared)

# for plotting
train = new_data[:3500]
valid = new_data[3500:]
pd.options.mode.chained_assignment = None
valid['Predictions'] = pm10

plt.figure(figsize=(16, 8))
plt.plot(train['PM10 (ug/m3)'])
[plt_1, plt_2] = plt.plot(valid[['PM10 (ug/m3)', 'Predictions']])
plt.legend([plt_1, plt_2], ['PM10 (ug/m3)', 'Predictions'])

x_ticks_length = np.arange(len(data['DateTime']))
plt.xticks(x_ticks_length[::30], data['DateTime'][::30], rotation=45)

plt.show()
plt.savefig('./output/predict_pm10_lstm.png', dpi=800)

"""
This is a very simple model

Please complete your assignment with a better one

refer to this tutorial

https://www.tensorflow.org/tutorials/structured_data/time_series

               LSTM - Result (lr=0.01  batch=4 epoch = 20)
	    count	 mean	std	 min	25% 	50%	    75%	    max
RMSE	 30	     0.33     ..... ..... ..... ..... 
MAE      30      0.1      ..... ..... ..... ..... ..... 
R-squre  30      0.1      ..... ..... ..... ..... ..... 

  
Epoch 1/20
1534/1534 [==============================] - 4s 3ms/step - loss: 0.3780 - mean_absolute_error: 0.3770 - val_loss: 0.0286 - val_mean_absolute_error: 0.1310
Epoch 2/20
1534/1534 [==============================] - 4s 3ms/step - loss: 0.0156 - mean_absolute_error: 0.0945 - val_loss: 0.0105 - val_mean_absolute_error: 0.0761
Epoch 3/20
1534/1534 [==============================] - 4s 3ms/step - loss: 0.0102 - mean_absolute_error: 0.0750 - val_loss: 0.0093 - val_mean_absolute_error: 0.0718
Epoch 4/20
1534/1534 [==============================] - 4s 3ms/step - loss: 0.0095 - mean_absolute_error: 0.0717 - val_loss: 0.0089 - val_mean_absolute_error: 0.0700
Epoch 5/20
1534/1534 [==============================] - 4s 2ms/step - loss: 0.0092 - mean_absolute_error: 0.0704 - val_loss: 0.0088 - val_mean_absolute_error: 0.0697
Epoch 6/20
1534/1534 [==============================] - 4s 2ms/step - loss: 0.0091 - mean_absolute_error: 0.0700 - val_loss: 0.0087 - val_mean_absolute_error: 0.0694
Epoch 7/20
1534/1534 [==============================] - 4s 2ms/step - loss: 0.0091 - mean_absolute_error: 0.0697 - val_loss: 0.0087 - val_mean_absolute_error: 0.0697
Epoch 8/20
1534/1534 [==============================] - 4s 3ms/step - loss: 0.0091 - mean_absolute_error: 0.0697 - val_loss: 0.0088 - val_mean_absolute_error: 0.0698
439/439 [==============================] - 1s 2ms/step - loss: 0.0088 - mean_absolute_error: 0.0698


plt.plot([0.3780, 0.0156, 0.0102, 0.0095, 0.0092, 0.0091, 0.0091, 0.0091, 0.0091, 0.0088])

plt.ylabel('loss per epoch')
plt.show()

"""
