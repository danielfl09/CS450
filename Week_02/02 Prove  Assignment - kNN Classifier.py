from numpy import sqrt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()


class KNearestClassifier:

    def __init__(self, neighbors, train_data=[], train_target=[]):
        self.neighbors = neighbors
        self.train_data = train_data
        self.train_target = train_target

    def fit(self, data, targets):
        # scalar = StandardScaler()
        # scalar.fit(data)
        self.train_data = data  # scalar.transform(data)
        self.train_target = targets
        return KNearestClassifier(self.neighbors, self.train_data, self.train_target)

    def predict(self, predict):
        distances = []
        vote_results = []

        for row in predict:

            for i in range(0, len(self.train_data)):
                sub_total = 0

                for j in range(0, len(self.train_data[i])):
                    sub_total += (self.train_data[i, j] - row[j]) ** 2
                distance = sqrt(sub_total)
                distances.append((distance, self.train_target[i]))

            ordered_distances = sorted(distances, key=lambda x: x[0])

            votes = []
            count = 0
            for key, value in ordered_distances:
                if count < self.neighbors:
                    votes.append(value)
                    count += 1
            vote_results.append(max(set(votes), key=votes.count))

        return vote_results


# Testing and Experimenting
kneighbor_accuracy = []
my_accuracy = []

for i in range(1, 6):
    data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.3, shuffle=True)


    classifier = KNearestClassifier(i)
    model = classifier.fit(data_train, target_train)
    prediction = model.predict(data_test)
    # print("Prediction: ", prediction)
    #print("Accuracy:", accuracy_score(target_test, prediction))
    my_accuracy.append((i, accuracy_score(target_test, prediction)))

    classifier = KNeighborsClassifier(n_neighbors=i)
    model = classifier.fit(data_train, target_train)
    prediction = model.predict(data_test)
    # print("Prediction: ", prediction)
    #print("Accuracy:", accuracy_score(target_test, prediction))
    kneighbor_accuracy.append((i, accuracy_score(target_test, prediction)))

print("my_accuracy: ", my_accuracy)
print("kneighbor_accuracy: ", kneighbor_accuracy)
