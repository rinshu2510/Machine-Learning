import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ad_data = pd.read_csv("advertising.csv")
# print(ad_data.head())
# print(ad_data.info())

#Exploring Data

sns.set_style('whitegrid')
# ad_data['Age'].hist(bins = 30)
# plt.xlabel('Age')
# plt.show()

# sns.jointplot(y='Area Income',x='Age',data=ad_data)
# plt.show()

# sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, color = 'red', kind="kde")
# plt.show()

# sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data, color='green', )
# plt.show()

# sns.pairplot(ad_data, hue= 'Clicked on Ad', palette='bwr')
# plt.show()

# Training the model

from sklearn.model_selection import train_test_split


x = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)

prediction = logmodel.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))

