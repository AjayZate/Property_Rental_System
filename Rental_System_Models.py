import pandas as pd
import sklearn

data = pd.read_csv('house_price.csv')
print(data.head())

# Describing data to get statistical details
print(data.describe())

# Checking for Null Values
print(data.isna().sum())

# Dropping irrelevant column
data.drop('Unnamed: 0', axis=1, inplace=True)

print(data.head())

# Mapping Numbers for Location
data['Location'] = data['Location'].map({'Whitefield': 1, 'Bommanahalli': 2})
print(data.head())

# Separating Dependent and independent features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(X.head())
print(y.head())

# Splitting data as Train and Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

DT = DecisionTreeRegressor()
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)

from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))

print(X_test.info())

import pickle

file = open('DT_Rental_System_Model.pkl', 'wb')

pickle.dump(DT, file)
