import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

corr_df = pd.read_csv("./output/clean_data.csv")

""" 
in your report, you may have to do a literature review, tell us what is it? 
and explain the correlation according to this heatmap
"""
correlations = corr_df.corr(method="pearson")  # Pearson or any other algorithms

plt.figure(figsize=(12, 10))

#                        yellow, green, blue, you may want to change the color
sns.heatmap(data=correlations, annot=True, cmap="YlGnBu")

plt.show()

plt.savefig('./output/st_albans_pearson_corr.png', dpi=800)

"""
feature selection algorithms:
all up to you

I don't want to restrict you too much.
"""

# TODO: feature selection and time lag (t-1 and t-2) Please do it on your own
#  so, let's randomly select some of these to complete the lab

random_ones_not_most_influential_features = corr_df[
    [
        "DateTime",
        "Relative humidity (%)",
        'Wind maximum (m/s)',
        'NO (ug/m3)',
        'NO2 (ug/m3)',
        'Temperature 2m (DegC)',
        'PM10 (ug/m3)'
    ]
]

random_ones_not_most_influential_features.to_csv("./output/cleaned_best_feature_data.csv", index=False)
