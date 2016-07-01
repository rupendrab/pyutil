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
