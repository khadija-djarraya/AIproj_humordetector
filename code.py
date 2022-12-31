import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.exceptions import NotFittedError
import argilla as rg
from modAL.models import ActiveLearner

# reading csv dataset, separator is a ',' by default
data = pd.read_csv("dataset.csv")


print(data.tail(5))  # printing the last 5 elements of the dataset.

cols = data.shape  # shape of the matrix that read the dataset
print('There are {} jokes.\n'.format(cols[0]))

print(data.info())  # shows further information on the dataset

y = data.humor  # this is our target, or "label", it is the humor column, a boolean that says whether a text is
# humorous or not.
x = data.drop('humor', axis=1)
train, test = train_test_split(data, test_size=0.2, random_state=45)
print(train.head())
print(train.shape)
print(test.shape)

# Define our classification model
classifier = MultinomialNB()

# Define active learner
learner = ActiveLearner(
    estimator=classifier,  # uses the classifier as an estimator of the most uncertain predictions
)

# The resulting matrices will have the shape of (`nr of examples`, `nr of word n-grams`)
vectorizer = CountVectorizer(ngram_range=(1, 5))

x_train = vectorizer.fit_transform(train.text)
x_test = vectorizer.transform(test.text)

#  Active Learning Loop :

# Number of instances we want to annotate per iteration
n_instances =10

# Accuracies after each iteration to keep track of our improvement
accuracies = []

# 1. Annotate Samples

# query examples from our training pool with the most uncertain prediction
query_idx, query_inst = learner.query(x_train, n_instances=n_instances)

# get predictions for the queried examples
try:
    probabilities = learner.predict_proba(x_train[query_idx])
# For the very first query we do not have any predictions
except NotFittedError:
    probabilities = [[0.5,0.5]] * n_instances

# Build the Argilla records
records = [
    rg.TextClassificationRecord(
        id=idx,
        text=train.text.iloc[idx],
        prediction=list(zip(["False","True"], probs)),
        # labelled funny or not funny according to the advice of Argilla creators
        prediction_agent="MultinomialNB",
    )
    for idx, probs in zip(query_idx, probabilities)
]

# Log the records
rg.log(records, name="active_learning_jokes")

# we switch over to the UI, where we can find the newly logged examples in the active_learning_jokes dataset.

# 2. Teach the learner

# Load the annotated records into a pandas DataFrame
records_df = rg.load("active_learning_jokes1", ids=query_idx.tolist()).to_pandas()

# check if all examples were annotated
if any(records_df.annotation.isna()):
    raise UserWarning(
        "Please annotate first all your samples before teaching the model"
    )

# train the classifier with the newly annotated examples
y_train = records_df.annotation.map(lambda x: int(x == "True"))
learner.teach(X=x_train[query_idx], y=y_train.to_list())

# Keep track of our improvement
accuracies.append(learner.score(X=x_test, y=test.humor))
print(accuracies)

# The training process constitutes of doing 1. and 2. on a loop multiple times.

# 3. Plotting Improvement
import matplotlib.pyplot as plt

# Plot the accuracy versus the iteration number
plt.plot(accuracies)
plt.xlabel("Number of iterations")
plt.ylabel("Accuracy")

plt.show()


