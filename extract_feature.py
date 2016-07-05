### one hot representation, categories columnarized as 1-0, one 
### column per type

from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()
instances = [
    {'city': 'New York'},
    {'city': 'San Francisco'},
    {'city': 'Chapel Hill'}
]
one_encoded = onehot_encoder.fit_transform(instances)
# dir(one_encoded)
# one_encoded.indices
# print(onehot_encoder.fit_transform(instances))
print(onehot_encoder.fit_transform(instances).toarray())

######

## Bag of words
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(corpus)
# print(counts)
print(counts.todense())
print(vectorizer.vocabulary_)

######

## Now compute distances
from sklearn.metrics.pairwise import euclidean_distances
print('Distance between 1st and 2nd document: %0.4f' % euclidean_distances(counts[0], counts[1]))
print('Distance between 1st and 3rd document: %0.4f' % euclidean_distances(counts[0], counts[2]))
print('Distance between 2nd and 3rd document: %0.4f' % euclidean_distances(counts[1], counts[2]))

######

## Stop word filtering
vectorizer = CountVectorizer(stop_words='english')
counts = vectorizer.fit_transform(corpus)
# print(counts)
print(counts.todense())
print(vectorizer.vocabulary_)
print('Distance between 1st and 2nd document: %0.4f' % euclidean_distances(counts[0], counts[1]))
print('Distance between 1st and 3rd document: %0.4f' % euclidean_distances(counts[0], counts[2]))
print('Distance between 2nd and 3rd document: %0.4f' % euclidean_distances(counts[1], counts[2]))

######

from sklearn.feature_extraction.text import CountVectorizer
corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
vectorizer = CountVectorizer(stop_words = 'english')
trans = vectorizer.fit_transform(corpus)
print(trans.todense())
print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names())
counters = [(v[0], trans.todense()[0,v[1]]) for v in vectorizer.vocabulary_.items()]
print(counters)
counters = [(word, trans.todense()[0,pos]) for (word,pos) in vectorizer.vocabulary_.items()]
print(counters)

######

## Use TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'The dog ate a sandwich and I ate a sandwich',
    'The wizard transfigured a sandwich'
]
vectorizer = TfidfVectorizer(stop_words='english')

print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.get_feature_names())

######

## Use hashing to prevent creating and maintaining a dictionary in memory
from sklearn.feature_extraction.text import HashingVectorizer
corpus = ['the', 'ate', 'bacon', 'cat']
vectorizer = HashingVectorizer(n_features=6)
print(vectorizer.transform(corpus).todense())

######

from sklearn import datasets
digits = datasets.load_digits()
print('Digit: ', digits.target[0])
print(digits.images[0])
print('Feature vector:\n', digits.images[0].reshape(-1,64))

######

from sklearn import preprocessing
import numpy as np
X = np.array(
    [
        [0., 0., 5., 13., 9., 1.],
        [0., 0., 13., 15., 10., 15.],
        [0., 3., 15., 2., 0., 11.]
    ]
)
X1 = preprocessing.scale(X)
print(X1)
