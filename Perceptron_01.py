from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron

######

### This one downloads a large dataset, so it may take a while
categories = ['rec.sport.hockey', 'rec.sport.baseball', 'rec.autos']
newsgroup_train = fetch_20newsgroups(subset='train', categories=categories, 
                                    remove=('headers', 'footers', 'quotes'))
newsgroup_test = fetch_20newsgroups(subset='test', categories=categories, 
                                    remove=('headers', 'footers', 'quotes'))
                                    
######

### Explore the data
for txt in (newsgroup_train.data)[0:2]:
    print(txt)
    print('.....')
print(newsgroup_train.target[0:2])
print(newsgroup_train.target_names)
print("No. of newsgroups: %d" % len(newsgroup_train.data))

######

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroup_train.data)
X_test = vectorizer.transform(newsgroup_test.data)

######

## Explore the Tfidf data
for txt in (X_train)[0:2]:
    print(txt)
    print('.....')

######

classifier = Perceptron(n_iter=100, eta0=0.1)
classifier.fit(X_train, newsgroup_train.target)
predictions = classifier.predict(X_test)
print(classification_report(newsgroup_test.target, predictions))
