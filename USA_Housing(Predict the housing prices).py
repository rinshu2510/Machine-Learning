import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('USA_Housing.csv')
print(df.head())
print(df.describe())

# sns.pairplot(df)
# sns.histplot(df['Price'])
# sns.heatmap(df.corr(),annot=True)
# plt.show()

# print(df.columns)
x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train,y_train)

# print(lm.intercept_)

cdf = pd.DataFrame(lm.coef_,x.columns,['Coeff'])
# print(cdf)

# Predictions
prediction = lm.predict(x_test)

# print(prediction)
# plt.scatter(y_test,prediction)
# sns.histplot(y_test-prediction)
# plt.show()

from sklearn import metrics

print(metrics.mean_absolute_error(y_test,prediction))
print(metrics.mean_squared_error(y_test,prediction))
print(np.sqrt(metrics.mean_squared_error(y_test,prediction)))