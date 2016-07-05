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

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
# print(predictions[:5])
# print(X_test_raw.head())
# print(type(df[1]), type(X_test_raw[0:0]))
for i, prediction in enumerate(predictions[:5]):
   # print(prediction, X_test_raw.tolist()[i])
   print('Prediction: %s. Message: %s' % (prediction, X_test_raw.tolist()[i]))

total, correct = reduce(
    lambda x, y: (x[0]+y[0], x[1]+y[1]),
    [(1, 1 if prediction == y_test.tolist()[i] else 0) for i, prediction in enumerate(predictions)]
    )
# print(total, correct)
# print(type(correct), type(total))
print("Percentage correct: %.4f" % (correct * 100 / total))

######
