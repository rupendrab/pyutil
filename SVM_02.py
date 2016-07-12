## Download data from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/#download

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from PIL import Image

######

def resize_and_crop(image, size):
    img_ratio = image.size[0] / float(image.size[1])
    ratio = size[0] / float(size[1])
    if ratio > img_ratio:
        image = image.resize((size[0],size[0] * image.size[1] / image.size[0]), 
                             Image.ANTIALIAS)
        image = image.crop((0,0,30,30))
    elif ratio < img_ratio:
        image = image.resize((size[1] * image.size[0] / image.size[1], size[1]), 
                             Image.ANTIALIAS)
        image = image.crop((0,0,30,30))
    else:
        image = image.resize((size[0], size[1]), Image.ANTIALIAS)
    return image

######

X = []
y = []

no_images = 0
imageDir = 'Chars74k\\English\\Img\\GoodImg\\Bmp\\'
for path, subdirs, files in os.walk(imageDir):
    for filename in files:
        f = os.path.join(path, filename)
        no_images += 1
        # print(no_images, f)
        img = Image.open(f).convert('L') # convert to grayscale
        img_resized = resize_and_crop(img, (30, 30))
        img_resized = np.asarray(img_resized.getdata(), dtype=np.float64) \
                    .reshape((img_resized.size[1] * img_resized.size[0],1))
        X.append(img_resized)
        target = filename[3:filename.index('-')]
        # print(target)
        y.append(target)
print("Number of images processed = %d" % no_images)

######

X = np.array(X)
print(X.shape[:2])
X = X.reshape(X.shape[:2])
print(X.shape)
print(X[0].shape)

######

classifier = SVC(verbose=0, kernel='poly', degree=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

######

classifier.fit(X_train, y_train)

######

predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
