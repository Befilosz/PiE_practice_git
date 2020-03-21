import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)
# reading boston dataset into pandas dataframe

y = boston.target # y as a target(price)
y2 = boston.target

df.head() # printing the head of the dataset

col = df[['LSTAT']]
col2 = df[['RM']]
image = col.hist(bins=100) # plotting two chosen columns
image2 = col2.hist(bins=100)


from sklearn.linear_model import LinearRegression
col_1 = df[['LSTAT']]
col_2 = df[['RM']]
reg1 = LinearRegression().fit(col_1, y) # fitting the columns against the target
reg2 = LinearRegression().fit(col_2, y2)


col_1_pred = reg1.predict(col_1) # predictions used later in plotting
col_2_pred = reg2.predict(col_2)

import matplotlib.pyplot as plt1 
import matplotlib.pyplot as plt2 

plt1.scatter(col_1, y, color='blue')
plt1.plot(col_1, col_1_pred, color='red', linewidth=2)
plt1.xlabel("LSTAT")
plt1.ylabel("Prices of the houses in 1000$'s")
plt1.title("Linear fitting LSTAT against the price")

plt1.show()

plt2.scatter(col_2, y2, color='blue')
plt2.plot(col_2, col_2_pred, color='red', linewidth=2)
plt2.xlabel("RM")
plt2.ylabel("Prices of the houses in 1000$'s")
plt2.title("Linear fitting RM against the price")
plt2.show()

