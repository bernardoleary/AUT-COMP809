import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# please use your own dataset
df = pd.read_csv("./dataset/St_Albans_Christchurch.csv")

df.hist(stacked=True, bins=10)
plt.subplots_adjust(hspace=0.5)
plt.show()
plt.savefig('./output/st_albans_histogram.png', dpi=800)

"""
Just some inspiration.
In order to get A+.
You may want to plot it in a better way
and interpret your plot in a better way

(Do a proper EDA (Exploratory data analysis )) !!
"""

df['year'] = pd.to_datetime(df['DateTime']).dt.strftime('%y')
df['month'] = pd.to_datetime(df['DateTime']).dt.strftime('%b')
df['week'] = pd.to_datetime(df['DateTime']).dt.strftime('%A')
df['hour'] = pd.to_datetime(df['DateTime']).dt.strftime('%H')

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
fig.set_figheight(12)
fig.set_figwidth(9)

sns.boxplot(x='year', y='PM10 (ug/m3)', data=df, ax=ax1)
sns.boxplot(x='month', y='PM10 (ug/m3)', data=df, ax=ax2)
sns.boxplot(x='week', y='PM10 (ug/m3)', data=df, ax=ax3)
sns.boxplot(x='hour', y='PM10 (ug/m3)', data=df, ax=ax4)

plt.show()
plt.savefig('./output/st_albans_pm_box_plot.png', dpi=800)

"""
Again, it is a very simple example.
If you want to get good marks, you have to explore the data in your own way.
"""

