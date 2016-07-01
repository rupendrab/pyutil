import pandas as pd
df = pd.read_csv('winequality-red.csv', sep=';')
df.describe()

######

%matplotlib inline

import matplotlib.pyplot as plt
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol against Quality')
plt.show()

######

plt.scatter(df['volatile acidity'], df['quality'])
plt.xlabel('Volatile Acidity')
plt.ylabel('Quality')
plt.title('Alcohol against Quality')
plt.show()

######

df.corr()

######

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
X = df[list(df.columns)[:-1]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print('R-Squared: %.4f' % regressor.score(X_test, y_test))

######

from sklearn.cross_validation import cross_val_score
regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)
print(scores.mean(), scores)

######

plt.scatter(y_test, y_predictions)
plt.xlabel('Quality')
plt.ylabel('Predicted Quality')
plt.title('Predicted Quality against True Quality')
plt.show()

######

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(np.array(y_train).reshape(-1,1))
X_test = X_scaler.fit_transform(X_test)
y_test = y_scaler.fit_transform(np.array(y_test).reshape(-1,1))


regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train[:,0], cv=5)
print('Cross validation r-squared loss: ', scores)
print('Average Cross validation r-squared loss: %.4f' % np.mean(scores))
regressor.fit(X_test, y_test[:,0])
print('Test r-squared score: %.4f' % regressor.score(X_test, y_test[:,0]))


