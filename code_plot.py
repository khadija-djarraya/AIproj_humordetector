#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.exceptions import NotFittedError
import argilla as rg
from modAL.models import ActiveLearner

# reading csv dataset, separator is a ',' by default

data = pd.read_csv('dataset.csv')

print (data.tail(5))  # printing the last 5 elements of the dataset.

cols = data.shape  # shape of the matrix that read the dataset
print ('There are {} jokes.\n'.format(cols[0]))

print (data.info())  # shows further information on the dataset

y = data.humor  # this is our target, or "label", it is the humor column, a boolean that says whether a text is

# humorous or not.

x = data.drop('humor', axis=1)
(train, test) = train_test_split(data, test_size=0.2, random_state=45)
print (train.head())
print (train.shape)
print (test.shape)

# Define our classification model

classifier = MultinomialNB()

# Define active learner

learner = ActiveLearner(estimator=classifier)  # uses the classifier as an estimator of the most uncertain predictions

# The resulting matrices will have the shape of (`nr of examples`, `nr of word n-grams`)

vectorizer = CountVectorizer(ngram_range=(1, 5))

X_train = vectorizer.fit_transform(train.text)
X_test = vectorizer.transform(test.text)

#  Active Learning Loop :

# Number of instances we want to annotate per iteration

n_instances = 10

# Accuracies after each iteration to keep track of our improvement

accuracies = []
import numpy as np

n_iterations = 150
n_instances = 10
random_samples = 50

# max uncertainty strategy

accuracies_max = []
for i in range(random_samples):
    train_rnd_df = train  # .sample(frac=1)
    test_rnd_df = test  # .sample(frac=1)
    X_rnd_train = vectorizer.transform(train_rnd_df.text)
    X_rnd_test = vectorizer.transform(test_rnd_df.text)
    print ('running')
    (accuracies, learner) = ([],
                             ActiveLearner(estimator=MultinomialNB()))

    for i in range(n_iterations):
        (query_idx, _) = learner.query(X_rnd_train,
                n_instances=n_instances)
        learner.teach(X=X_rnd_train[query_idx],
                      y=train_rnd_df.humor.iloc[query_idx].to_list())
        accuracies.append(learner.score(X=X_rnd_test,
                          y=test_rnd_df.humor))
    accuracies_max.append(accuracies)

# max margin strategy
from modAL.uncertainty import margin_sampling
accuracies_mar= []
for i in range(random_samples):
    train_rnd_df = train  # .sample(frac=1)
    test_rnd_df = test  # .sample(frac=1)
    X_rnd_train = vectorizer.transform(train_rnd_df.text)
    X_rnd_test = vectorizer.transform(test_rnd_df.text)
    print ('running')
    (accuracies, learner) = ([],
                             ActiveLearner(estimator=MultinomialNB(), query_strategy=margin_sampling))

    for i in range(n_iterations):
        (query_idx, _) = learner.query(X_rnd_train,
                n_instances=n_instances)
        learner.teach(X=X_rnd_train[query_idx],
                      y=train_rnd_df.humor.iloc[query_idx].to_list())
        accuracies.append(learner.score(X=X_rnd_test,
                          y=test_rnd_df.humor))
    accuracies_mar.append(accuracies)

# max margin strategy
from modAL.uncertainty import entropy_sampling
accuracies_ent = []
for i in range(random_samples):
    train_rnd_df = train  # .sample(frac=1)
    test_rnd_df = test  # .sample(frac=1)
    X_rnd_train = vectorizer.transform(train_rnd_df.text)
    X_rnd_test = vectorizer.transform(test_rnd_df.text)
    print ('running')
    (accuracies, learner) = ([],
                             ActiveLearner(estimator=MultinomialNB(), query_strategy=entropy_sampling))

    for i in range(n_iterations):
        (query_idx, _) = learner.query(X_rnd_train,
                n_instances=n_instances)
        learner.teach(X=X_rnd_train[query_idx],
                      y=train_rnd_df.humor.iloc[query_idx].to_list())
        accuracies.append(learner.score(X=X_rnd_test,
                          y=test_rnd_df.humor))
    accuracies_ent.append(accuracies)



# random strategy

accuracies_rnd = []
for i in range(random_samples):
    (accuracies, learner) = ([],
                             ActiveLearner(estimator=MultinomialNB()))

    for random_idx in np.random.choice(X_train.shape[0],
            size=(n_iterations, n_instances), replace=False):
        learner.teach(X=X_train[random_idx],
                      y=train.humor.iloc[random_idx].to_list())
        accuracies.append(learner.score(X=X_test, y=test.humor))
    accuracies_rnd.append(accuracies)

(arr_max, arr_rnd,arr_mar,arr_ent) = (np.array(accuracies_max),
                      np.array(accuracies_rnd),
                      np.array(accuracies_mar),
                      np.array(accuracies_ent))

import matplotlib.pyplot as plt

plt.plot(range(n_iterations), arr_max.mean(0))
#plt.fill_between(range(n_iterations), arr_max.mean(0) - arr_max.std(0),
#                 arr_max.mean(0) + arr_max.std(0), alpha=0.2)
plt.plot(range(n_iterations), arr_rnd.mean(0))
#plt.fill_between(range(n_iterations), arr_rnd.mean(0) - arr_rnd.std(0),
#                 arr_rnd.mean(0) + arr_rnd.std(0), alpha=0.2)
plt.plot(range(n_iterations), arr_mar.mean(0))
#plt.fill_between(range(n_iterations), arr_mar.mean(0) - arr_mar.std(0),
#                 arr_mar.mean(0) + arr_mar.std(0), alpha=0.2)
plt.plot(range(n_iterations), arr_ent.mean(0))
#plt.fill_between(range(n_iterations), arr_ent.mean(0) - arr_ent.std(0),
#                 arr_ent.mean(0) + arr_ent.std(0), alpha=0.2)



plt.xlim(0, 15)
plt.title('Sampling strategies: Max uncertainty vs random')
plt.xlabel('Number of annotation iterations')
plt.ylabel('Accuracy')
plt.legend(['max uncertainty', 'random sampling','margin uncertainty','entropy sampling'], loc=4)
plt.show()
