import pandas as pd
from functools import reduce
df = pd.read_csv('SMSSpamCollection', delimiter='\t', header=None)
print(df.head())
print(df.size)
# print(reduce(lambda x, y: x+y, range(1,101)))
# print([(1 if line == "ham" else 0, 1 if line =="spam" else 0) for line in df[0]])
(ham, spam) = reduce(lambda x,y: (x[0]+y[0],x[1]+y[1]),[(1 if line == "ham" else 0, 1 if line == "spam" else 0) for line in df[0]])
print(counts)
print('Number of spam messages: %d' % spam)
print('Number of ham messages: %d' % ham)
print('Number of spam messages: %d' % df[df[0] == "spam"][0].count())
print('Number of ham messages: %d' % df[df[0] == "ham"][0].count())

######

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import preprocessing

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

lb = preprocessing.LabelBinarizer()
# print(y_train.tolist())
lb.fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)
# print(y_test)
classifier = LogisticRegression()
classifier.fit(X_train, y_train.ravel())
predictions = classifier.predict(X_test)
# print(predictions[:5])
# print(X_test_raw.head())
# print(type(df[1]), type(X_test_raw[0:0]))
for i, prediction in enumerate(predictions[:5]):
   # print(prediction, X_test_raw.tolist()[i])
   print('Prediction: %s. Message: %s' % (prediction, X_test_raw.tolist()[i]))

total, correct = reduce(
    lambda x, y: (x[0]+y[0], x[1]+y[1]),
    [(1, 1 if prediction == y_test[i] else 0) for i, prediction in enumerate(predictions)]
    )
# print(total, correct)
# print(type(correct), type(total))
print("Percentage correct: %.4f" % (correct * 100 / total))

######

%matplotlib inline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test.tolist(), predictions)
print(conf_matrix)
plt.matshow(conf_matrix)
# plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(2)
labels = sorted(["spam", "ham"])
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

## Prediced spam actual spam
prediction_stats = []
for i in [0,1]:
    prediction_stats.append([])
    for j in [0,1]:
        prediction_stats[i].append(0)

# print(prediction_stats)
def summarize(x, y):
    global prediction_stats
    # print(x,y)
    prediction_stats[x][y] = prediction_stats[x][y] + 1
    1
    
spamToSpam = [summarize(y_test[i][0], prediction) for i, prediction in enumerate(predictions)]
print(prediction_stats)

######

## Accuracy, Precision and Recall
from sklearn.metrics  import accuracy_score

y_pred, y_true = [0,1,1,0], [1,1,1,1]
print('Accuracy:', accuracy_score(y_true, y_pred))
print('Accuracy of spam/ham', accuracy_score(y_test, predictions))
"""
print(X_train.shape)
print(y_train.shape)
print(X_train.todense())
print(y_train)
print(y_train.ravel())
"""
scores = cross_val_score(classifier, X_train, y_train.ravel(), cv=5)
print('Accuracy', np.mean(scores), scores)

precisions = cross_val_score(classifier, X_train, y_train.ravel(), cv=5, scoring='precision')
print('Precision', np.mean(precisions), precisions)

recalls = cross_val_score(classifier, X_train, y_train.ravel(), cv=5, scoring='recall')
print('Recall', np.mean(recalls), recalls)
