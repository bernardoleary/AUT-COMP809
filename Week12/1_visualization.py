import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

# please use your own dataset
data = pd.read_csv("./dataset/St_Albans_Christchurch.csv")
data["DateTime"] = pd.to_datetime(data["DateTime"]).dt.strftime('%Y-%m-%d %H:%M:%S')

print('Number of rows and columns:', data.shape)

print(data.head(5))
# Save it to CSV, the copy the tabular data to your report
data.head(5).to_csv("./output/head_5.csv")

print(data.describe().T)
# Save it to CSV, the copy the tabular data to your report
data.describe().T.to_csv("./output/data_summary.csv")

# Visualising the results
fig, ax = plt.subplots()

ax.plot(data["DateTime"], data["PM10 (ug/m3)"], color='blue', label='PM_10')

# please use your own dataset
ax.set_title('St. Albans PM_10')
ax.set_xlabel('Date')
ax.set_ylabel('PM10 (ug/m3)')
ax.legend()

# set up figure size
fig.set_figheight(9)
fig.set_figwidth(16)

# x-tick frequency 100
# format the date to YYYY-MM
x_ticks_length = np.arange(len(data['DateTime']))
plt.xticks(x_ticks_length[::100],
           pd.to_datetime(data['DateTime'][::100]).dt.strftime('%Y-%m-%d'),
           rotation=45)

"""
I am using an random number,
 don't know what is the NZ standard and WHO standard
 every 24 hour or every hour?
find out by yourselves
"""
ax.axhline(y=50, xmin=0, xmax=4000, color='r')

plt.show()
plt.savefig('./output/st_albans_pm_plot.png', dpi=800)
