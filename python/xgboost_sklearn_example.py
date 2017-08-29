# coding=utf-8
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# load data
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
# y - right column
Y = dataset[:,8]

# split data into train and test sets
seed = 7  # in real life - must be random. set same seed for reproducible results
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


def try_classifier(model, name):
    print "\n{}:".format(name)
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    #round predictions
    predictions = [round(value) for value in y_pred]

    #prediction table
    print(map(lambda (x, y): (x + 1, y), enumerate(predictions)))

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    auc_score = roc_auc_score(y_test, predictions)
    print("Area under curve: %.2f%%" % (auc_score * 100.0))

try_classifier(XGBClassifier(), 'XGBoost')
try_classifier(MLPClassifier(), 'Multilayer perceptron')
try_classifier(LogisticRegression(), 'Logistic Regression')
