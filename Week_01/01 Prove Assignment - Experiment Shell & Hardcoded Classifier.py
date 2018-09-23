from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
type(iris)

Iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

data_train, data_test, target_train, target_test = train_test_split(Iris, Iris.target, test_size=0.3, shuffle=True)

classifier = GaussianNB()
model = classifier.fit(data_train, target_train)
type(model)

targets_predicted = model.predict(data_test)
len(targets_predicted)
len(target_test)


def Compare(x, y):
    count = 0

    for a, b in zip(x, y):
        if a == b:
            count += 1

    return count / len(x)


results = Compare(target_test, targets_predicted)

print(results)


class HardCodedModel:

    def __init__(self, x=[], y=[]):
        self.x = x
        self.y = y

    def predict(self, x):
        return_values = []

        for value in x:
            return_values.append(0)
        return return_values


class HardCodedClassifier:
    def __init__(self):
        pass

    def fit(self, x, y):
        return HardCodedModel(x, y)

classifier_2 = HardCodedClassifier()
model_2 = classifier_2.fit(data_train, target_train)
targets_predicted_2 = model_2.predict(data_test)

results_2 = Compare(target_test, targets_predicted_2)
print(results_2)
