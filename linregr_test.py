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

