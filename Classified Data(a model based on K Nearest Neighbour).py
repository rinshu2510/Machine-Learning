import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Classified Data.csv', index_col=0)
# print(df.head())
# print(df.info())

from  sklearn.preprocessing import StandardScaler

scaler =  StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
# print(scaled_features)
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
# print(df_feat)

from sklearn.model_selection import train_test_split

x = df_feat
y = df['TARGET CLASS']

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,Y_train)

pred = knn.predict(X_test)

# print(pred)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,pred))
print(confusion_matrix(Y_test,pred))

# Finding value of k using elbow method

error_rate = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,50), error_rate, linestyle='dashed',marker='o',markerfacecolor='red', markersize= 10)
plt.title('K Value vs Error Rate')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
print(classification_report(Y_test,pred))
print(confusion_matrix(Y_test,pred))
