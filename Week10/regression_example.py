import matplotlib.pyplot as plt
import pandas as pd
from fastai.tabular.core import add_datepart
from sklearn.linear_model import LinearRegression
import numpy as np

pd.options.mode.chained_assignment = None

data = pd.read_csv("TSLA.csv")
print('Number of rows and columns:', data.shape)
data.head(5)

# creating a separate dataset
new_data = data[['Date', 'Close']]

# new_data['Predictions'] = 0

"""
Let's say, today is 
2021,09,30

and you want to predict the stock price of TESLA on 2022,01,01

so you know the value of : Open,High,Low,Adj Close,Volume ?
you can't know it, right?

So, it means you don't have applicable features.
The only feature you can use, is the timestamp.

Thus, we have to create some other features from the timestamp.
"""

add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp

"""
This creates features such as:

‘Year’, ‘Month’, ‘Week’, ‘Day’, ‘Dayofweek’, ‘Dayofyear’, 
‘Is_month_end’, ‘Is_month_start’, ‘Is_quarter_end’, ‘Is_quarter_start’,  
‘Is_year_end’, and  ‘Is_year_start’.

Note: I have used add_datepart from fastai library. 
If you do not have it installed, 
you can simply use the command pip install fastai. 
Otherwise, you can create these feature using simple for loops in python. 
I have shown an example below.

Apart from this, 
we can add our own set of features 
that we believe would be relevant for the predictions.
 For instance, 
 my hypothesis is that the first and last days of the week 
 could potentially affect the closing price of the stock 
 far more than the other days.
  So I have created a feature
   that identifies whether a 
    given day is Monday/Friday
    or Tuesday/Wednesday/Thursday. 
This can be done using the following lines of code:


new_data['mon_fri'] = 0
for i in range(0, len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0
        
"""

# split into train and validation
train = new_data[:987]
valid = new_data[987:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

model = LinearRegression()
model.fit(x_train, y_train)

# make predictions and find the rmse
preds = model.predict(x_valid)
rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
print(rms)

# for plotting
# valid.loc[0, valid.columns.get_loc('Predictions')] = preds
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = preds

plt.figure(figsize=(16, 8))
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])

# Let's save the file, and insert it to your report!!
plt.savefig('./output/TESLA_regression_plot.png', dpi=800)
