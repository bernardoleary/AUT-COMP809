import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# please use your own dataset
df = pd.read_csv("./dataset/St_Albans_Christchurch.csv")
print(df.shape)

# 1. we may want to drop these columns with too many missing values.
#    in this example, let's drop the columns less than 20% values.
df_clean = df.dropna(thresh=df.shape[0] * .20, axis=1)

df_clean.to_csv("./output/clean_data.csv")

"""
 2. clean up the attribute-specific information about outliers and missing data
      a. you have multiple choices, like, replace the humidity above 100 to 100
      b. or you replace it to nearest non-null value 
             because the data is collect hourly, some attributes may not change rapidly in an hour
      c. or you can also replace the error value to the average value.
      
 All the strategies are acceptable, 
 as long as you can explain why you use the strategy
 show us your supporting evidences
 
"""
# example one: replace with a constant number
df_clean.loc[df_clean["Relative humidity (%)"] > 100, 'Relative humidity (%)'] = 100

# example two: replace with a mean
df_clean['NO2 (ug/m3)'].fillna((df_clean['NO2 (ug/m3)'].mean()), inplace=True)

# example three: replace with a median
df_clean['NO (ug/m3)'].fillna((df_clean['NO (ug/m3)'].median()), inplace=True)

# df_clean = df_clean.fillna(method='ffill')

"""
Remember, I only show your very simple examples.
You will need to check all the columns, you should tidy it up.
"""

"""
Finally!
Remember to save it in another file.
Then you don't have to do redo the preprossing time after times.

you'll read the clean data, then do the data mining! 
"""
df_clean.to_csv("./output/clean_data.csv", index=False)

df_1 = pd.read_csv("./output/clean_data.csv")

print(df_1.shape)
# Save it to CSV, the copy the tabular data to your report
df_1.describe().T.to_csv("./output/cleaned_data_summary.csv")
