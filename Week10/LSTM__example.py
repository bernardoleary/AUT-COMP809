import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow_addons.metrics import RSquare

data = pd.read_csv("TSLA.csv")
print('Number of rows and columns:', data.shape)
data.head(5)

# creating a separate dataset
new_data = data[['Date', 'Close']]
# reset index to the timestamp, instead of numeric sequential values like : 1,2,3,4
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

# creating train and test sets
dataset = new_data.values

train = dataset[0:987, :]
valid = dataset[987:, :]

# It’s a good idea to normalize the data before model fitting.
# This will boost the performance. You can read more here for the Min-Max Scaler:
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

"""

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
time_step = 60  # step = 100 ? 600 ? whatever

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
closing_price = model.predict(X_test)
# because we have scaled the data, so we should inverse the scale
closing_price = scaler.inverse_transform(closing_price)

# we can get the error, in this example, it is mean_absolute_error:
# print('Mean Absolute Error:', model.evaluate(X_test))

# you can also calculate the error by formula, such as rmse
rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
print('rmse:', rms)

correlation_matrix = np.corrcoef(
    np.reshape(valid, valid.shape[0]),
    np.reshape(closing_price, closing_price.shape[0])
)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy ** 2
print('RSquare:', r_squared)

"""
 stock price is affected by the news about the company
  and other factors like demonetization or merger/demerger of the companies.
  There are certain intangible factors as well which can often be impossible to predict beforehand.
"""

# for plotting
train = new_data[:987]
valid = new_data[987:]
pd.options.mode.chained_assignment = None
valid['Predictions'] = closing_price

plt.figure(figsize=(16, 8))
plt.plot(train['Close'])
[plt_1, plt_2] = plt.plot(valid[['Close', 'Predictions']])
plt.legend([plt_1, plt_2], ['Close', 'Predictions'])

x_ticks_length = np.arange(len(data['Date']))
plt.xticks(x_ticks_length[::30], data['Date'][::30], rotation=45)

plt.show()
plt.savefig('./output/TESLA_LSTM_plot.png', dpi=800)

"""
In the lab,
we don't have time to run a complex model.
If you are using CUDA-enabled graphics cards (such as 2070 super)
This may be finish in 5 minutes. 
if your GPU is not very good, it may require 10 minutes

model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))
# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)
"""
