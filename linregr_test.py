%matplotlib notebook

import matplotlib.pyplot as plt
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
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

