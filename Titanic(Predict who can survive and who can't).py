import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def impute_age(cols):
    Age = cols[0]
    Pclass = [1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')

print(train.head())
print(train.info())
print(test.head())
print((test.info()))

# Exploring the data for visualization

# To check how many null values are there ans in which column
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='BrBG')

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)
sns.countplot(x='Survived', hue='Sex', data=train, palette='BrBG')
sns.countplot(x='Survived', hue='Pclass', data=train)
sns.displot(train['Age'].dropna())
train['Age'].dropna().hist(bins=35)

sns.countplot(x='SibSp', data=train)

train['Fare'].hist(bins=40, figsize=(10, 4))
plt.show()

# cleaning our data

# checking average age
plt.figure(figsize=(10, 7))
sns.boxplot(x='Pclass', y='Age', data=train)
plt.show()

# placing Average age in place of null age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis=1)

print(train['Age'].isnull().value_counts())
print(test['Age'].isnull().value_counts())

# cabin columns

train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
print(train)
print(test)
sns.heatmap(data=train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.heatmap(data=test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

train.dropna(inplace=True)
sns.heatmap(data=train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

test.dropna(inplace=True)
sns.heatmap(data=test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

sex = pd.get_dummies(train['Sex'], drop_first=True)
Embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, Embark], axis=1)

sex = pd.get_dummies(test['Sex'], drop_first=True)
Embark = pd.get_dummies(test['Embarked'], drop_first=True)
test = pd.concat([test, sex, Embark], axis=1)

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], inplace=True, axis=1)
test.drop(['Sex', 'Embarked', 'Name', 'Ticket'], inplace=True, axis=1)

train.drop('PassengerId', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)

print(train.head())
print(train.info())
print(test.head())
print(test.info())

# Train and use a model to predict the classes of whether or not a passenger survived on titanic

from sklearn.model_selection import train_test_split

x = train.drop('Survived', axis=1)
y = train['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
prediction = logmodel.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(x_train, y_train)
prediction = rfc.predict(x_test)
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

# now we can run our model on real data knowing that which one is good model and which is working good

x_train = train.drop('Survived', axis=1)
y_train = train['Survived']
x_test = test

logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
prediction = logmodel.predict(x_test)

print(prediction)
