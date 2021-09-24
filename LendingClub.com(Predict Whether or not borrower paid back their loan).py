import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns

loans = pd.read_csv('loan_data.csv')
# print(loans.head())
print(loans.info())
# print((loans.describe()))

# Exploring Data Analysis

# plt.figure(figsize=(10,6))
# loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='credit.policy=1')
# loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='credit.policy=0')
# plt.legend()
# plt.xlabel('FICO')
# plt.show()


# plt.figure(figsize=(10,6))
# loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='not.fully.paid=1')
# loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='not.fully.paid=0')
# plt.legend()
# plt.xlabel('FICO')
# plt.show()

# sns.countplot(data=loans,x='purpose',hue='not.fully.paid')
# sns.jointplot(data=loans,x='fico',y='int.rate')
# plt.show()

# sns.lmplot(data=loans,x='fico',y='int.rate',hue='credit.policy',col='not.fully.paid',palette='Set1')
# plt.show()

# setting up the data

cat_feats = ['purpose']

final_data = pd.get_dummies(data=loans,columns=cat_feats,drop_first=True)
# print(final_data)

from sklearn.model_selection import train_test_split

x = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)
prediction = dtree.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)
prediction = rfc.predict(x_test)

print(classification_report(y_test,prediction))
print((confusion_matrix(y_test,prediction)))