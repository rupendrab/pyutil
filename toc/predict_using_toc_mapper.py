#!/usr/bin/env python3.5

import pandas as pd
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import pickle

from extract_toc import parseargs

def read_model_file(modelfile):
  with open(modelfile, 'rb') as f:
    sorted_y, vectorizer, le, grid_search = pickle.load(f)
  return (sorted_y, vectorizer, le, grid_search)

def binarysearch(sorted_y, val):
  ylen = len(sorted_y)
  n = int(ylen / 2)
  start = 0
  end = ylen - 1
  while start >= 0 and end < ylen and start < ylen and end >= 0 and start < end:
    n = int((start + end + 1) / 2)
    # print(start, end, n)
    chk = sorted_y[n]
    if (val == chk):
      return True
    elif (val > chk):
       start = n + 1
    else:
       end = n - 1
  return False

def get_topic(input, sorted_y, vectorizer, le, grid_search):
  if binarysearch(sorted_y, input):
    return (input, "actual")
  else:
    X_test_transformed = vectorizer.transform([input])
    # print(X_test_transformed)
    predicted = grid_search.predict(X_test_transformed)
    # print(predicted)
    predicted_val = le.inverse_transform(predicted[0])
    return (predicted_val, "guess")

if __name__ == '__main__':
  import sys
  args = sys.argv[1:]
  argsmap = parseargs(args)
  modelfile = argsmap.get("model")
  if (not modelfile):
    print('Model must be specified...')
    sys.exit(1)
  modelfile = modelfile[0]
  (sorted_y, vectorizer, le, grid_search) = read_model_file(modelfile)
  # for i, val in enumerate(sorted_y):
  #   print("%d\t%s" % (i, val))
  print(get_topic('Examination Conclusions and Comments', sorted_y, vectorizer, le, grid_search))
  print(get_topic('Examiner '' s Comments and Conclusions', sorted_y, vectorizer, le, grid_search))
  print(get_topic('Level 2/Medium Severity Violations', sorted_y, vectorizer, le, grid_search))

