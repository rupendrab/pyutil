%matplotlib notebook

import matplotlib.pyplot as plt
import numpy as np
X = np.array([[6], [8], [10], [14], [18]])
y = np.array([[7], [9], [13], [17.5], [18]])
plt.figure("Test")
plt.title("Pizza prices plotted against diameter")
plt.xlabel("Diamater in inches")
plt.ylabel("Price in dollars")
plt.plot(X, y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
print("A 12"" inch pizza should cost: $%.2f" % model.predict([[12]])[0])
print(model.coef_)
print(model.intercept_)
print('Residual sum of squares: %.2f' % np.mean((model.predict(X) - y) ** 2))

var_X = np.var(X.reshape(1,-1), ddof=1)
print(var_X)
cov_Xy = np.cov(X.reshape(1,-1), y.reshape(1,-1))[0][1]
print(cov_Xy)
## R-Square
X_test = [[8], [9], [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]
print('R-squared: %.4f' % model.score(X_test, y_test))

from numpy.linalg import inv
from numpy import dot, transpose

X = [[1,6,2],[1,8,1],[1,10,0],[1,14,2],[1,18,0]]
y = [[7], [9], [13], [17.5], [18]]
coeffs = dot(inv(dot(transpose(X), X)), dot(transpose(X), y))
print(coeffs)

from numpy.linalg import lstsq
print(lstsq(X, y)[0])

X = [[6,2],[8,1],[10,0],[14,2],[18,0]]
y = [[7],[9],[13],[17.5],[18]]
model = LinearRegression()
model.fit(X, y)
X_test = [[8,2],[9,0],[11,2],[16,2],[12,0]]
y_test = [[11],[8.5],[15],[18],[11]]
predictions = model.predict(X_test)
for i, prediction in enumerate(predictions):
  print('Predicted: %s, Target: %s' % (prediction, y_test[i]))
print('R squared: %.4f' % model.score(X_test, y_test))

