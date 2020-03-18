import numpy as np
import pandas as pd
from sklearn.datasets import load_boston 
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df.head()
col = df[['INDUS', 'PTRATIO']]
image = col.hist(bins=3)
