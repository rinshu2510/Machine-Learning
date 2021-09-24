import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def spec(num):
    if num == 0:
        return 'setosa'
    elif num ==1:
        return 'versicolor'
    else:
        return 'virginica'


iris = load_iris()

# print(iris)
# print(iris.keys())
# print(iris['DESCR'])

df = pd.DataFrame(iris['data'],columns=iris['feature_names'])
df['species'] = iris['target']
df['species'] = df['species'].apply(spec)
print(df)

# sns.pairplot(data=df,hue ='species',palette='Dark2')
# plt.show()

setosa = df[df['species']=='setosa']
# print(setosa)

sns.set_style('whitegrid')
# sns.kdeplot(data=setosa,y='sepal length (cm)',x='sepal width (cm)',cmap='plasma',shade=True, shade_lowest=False)
# plt.show()

from sklearn.model_selection import  train_test_split

x = df.drop('species',axis=1)
y = df['species']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=101)

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(x_train,y_train)
prediction = svc_model.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))

from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}

grid = GridSearchCV(SVC(),param_grid,verbose=2)

grid.fit(x_train,y_train)

print(grid.best_params_)
print(grid.best_estimator_)

svc_model = SVC(gamma=0.1)
svc_model.fit(x_train,y_train)
prediction = svc_model.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))

