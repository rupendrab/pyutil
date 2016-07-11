## Covariance Matrix

import numpy as np
X = [
    [2, 0, -1.4],
    [2.2, 0.2, -1.5],
    [2.4, 0.1, -1],
    [1.9, 0, -1.2]
    ]
print(np.array(X).T)
print(np.cov(np.array(X).T))

######

## Eigenvectors and Eigenvalues
X1 = np.array([
        [1,-2],
        [2, -3]
    ])
w, v = np.linalg.eig(X1)
print("w:", w)
print("v:", v)
left = X1.dot(v)
right = w * v
print(left - right)

######

## Sample Principal Component Analysis using eigenvalue decomposition

X2 = np.array([
        [0.9, 1],
        [2.4, 2.6],
        [1.2, 1.7],
        [0.5, 0.7],
        [0.3, 0.7],
        [1.8, 1.4],
        [0.5, 0.6],
        [0.3, 0.6],
        [2.5, 2.6],
        [1.3, 1.1]
    ])
X2T = X2.T
cov_matrix = np.cov(X2T)
print(cov_matrix)
W, V = np.linalg.eig(cov_matrix)
print('W (eigenvalues):')
print(W)
print('V (eigenvector):')
print(V)
avgs = X2.mean(0)
X2_1 = np.array([v - avgs[i] for row in X2 for i,v in enumerate(row)])
X2_minus_mean = X2_1.reshape(-1,len(avgs))
# print(X2_minus_mean)
# print(V[:,0])
pca_1 = X2_minus_mean.dot(V[:,0])
print('PCA 1')
print(pca_1)

######

## PCS for IRIS data

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
y = data.target
X = data.data
# print(X)
# print(y)
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

######

%matplotlib inline

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_X)):
    if (y[i] == 0):
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif (y[i] == 1):
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()

######

### Face recognition with PCA

from os import walk, path
import numpy as np
import mahotas as mh
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from PIL import Image

######

X = []
y = []
start_dir = "8365OS_Final code\\8365OS_07_Codes\\"
walk_dir = start_dir + "data\\att-faces\\orl_faces\\"
for dir_path, dir_names, file_names in walk(walk_dir):
    for fn in file_names:
        if fn[-3:] == 'pgm':
            image_filename = path.join(dir_path, fn)
            print(image_filename)
            imf = Image.open(image_filename)
            newname = image_filename + ".bmp"
            imf.save(newname)
            X.append(scale(mh.imread(newname, as_grey=True).reshape(10304).
                    astype('float32')))
            y.append(dir_path)
X = np.array(X)

######

## Use PCA to reduce the number of dimensions to 150 from 10304
print(X.shape)
print(len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA(n_components=150)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
print(X_train_reduced.shape, X_test_reduced.shape)
print('The original dimensions of the training data', X_train.shape)
print('The reduced dimensions of the training data', X_train_reduced.shape)

######

## Now use a Logistic Regression to check prediction accuracy
classifier = LogisticRegression()
accuracies = cross_val_score(classifier, X_train_reduced, y_train)
print('Cross validation accuracy:', np.mean(accuracies), accuracies)
classifier.fit(X_train_reduced, y_train)
predictions = classifier.predict(X_test_reduced)
print(classification_report(y_test, predictions))

######

## The below was used for testing. mahotas imread was not able to read
## a .pgm file. So we used the Pillows library to convert the image to 
## .bmp format first

%matplotlib inline
from PIL import Image
from pylab import imshow, show

im = Image.open(walk_dir + "/s1/9.pgm")
im.save('1.bmp')
x1 = mh.imread('1.bmp', as_grey=True).reshape(10304).astype('float32')
print(x1.shape)
scale(x1)
imshow(mh.imread('1.bmp', as_grey=True))
show()
