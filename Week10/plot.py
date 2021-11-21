import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("TSLA.csv")
print('Number of rows and columns:', df.shape)
df.head(5)

# Visualising the results
plt.figure(figsize=(16, 8))
plt.plot(df["Date"], df["Close"], color='blue', label='Close Price history')
plt.title('TESLA - Close Price history')
plt.xlabel('Time')
plt.ylabel('TESLA Stock Price')
plt.legend()
# plt.show()
plt.savefig('./output/TESLA_Stock_plot.png', dpi=800)
