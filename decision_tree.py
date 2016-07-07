### Find data from: 
## http://archive.ics.uci.edu/ml/datasets/Internet+Advertisements

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

######

df = pd.read_csv('ad.data', header=None)
# df = pd.read_csv('ad.data', header=None, na_values='?')

######

explanatory_variable_columns = set(df.columns.values)
response_variable_column = df[len(df.columns.values)-1]
explanatory_variable_columns.remove(len(df.columns.values)-1)
y = [1 if e == 'ad.' else 0 for e in response_variable_column]
X = df[list(explanatory_variable_columns)]

######

X = X.replace(to_replace=' *\?', value=-1, regex=True, inplace=False)
X = X.replace(to_replace=' *', value='', regex=True, inplace=False)

######

print(len(X.columns))
for i in range(len(X.columns)):
    X[i] = X[i].astype(float)
    
######

# print(set(X[3]))

X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline = Pipeline([
        ('clf', DecisionTreeClassifier(criterion='entropy'))
    ])
parameters = {
    'clf__max_depth': (150, 155, 160),
    'clf__min_samples_split': (1,2,3),
    'clf__min_samples_leaf': (1,2,3)
}

######

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)

######

print('Best score: %0.3f' % grid_search.best_score_)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" %(param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test)
print(classification_report(y_test, predictions))

######
