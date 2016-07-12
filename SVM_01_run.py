import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import matplotlib.cm as cm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

# digits = fetch_mldata('MNIST original', data_home='C:/Users/rubandyopadhyay/Downloads/data/mnist').data
# print(digits.shape)

if __name__ == '__main__':
    datadir = 'C:/Users/rubandyopadhyay/Downloads/data/mnist'
    data = fetch_mldata('MNIST original', data_home=datadir)
    X, y = data.data, data.target
    print(X.shape)
    X = X / 255.0 * 2 - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(X_train.shape)
    pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
    ])
    parameters = {
        'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
        'clf__C': (0.1, 0.3, 1, 2, 10, 30)
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=2, verbose=1,
                              scoring='accuracy')
    grid_search.fit(X_train[:10000], y_train[:10000])
    print('Best score: %0.3f' % grid_search.best_score_)
    print('Best parameter set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    predictions = grid_search.predict(X_test)
    print(classification_report(y_test, predictions))

