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
