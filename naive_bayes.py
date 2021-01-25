import numpy as np
from sklearn.naive_bayes import MultinomialNB

class NaiveBayes:
    def __init__(self, classes):
        self.classes = classes
        self.models = {}
        for item in self.classes:
            model = MultinomialNB()
            self.models[item] = model

    def fit(self, data_train, labels_train):
        for cls in self.classes:
            labels = labels_train[cls]
            self.models[cls].fit(data_train, labels)

    def predict(self, data_test):
        result = np.zeros((data_test.shape[0], len(self.classes)))
        for index, cls in enumerate(self.classes):
            result[:, index] = self.models[cls].predict(data_test)
        return result

    def predict_prob(self, data_test):
        result = np.zeros((data_test.shape[0], len(self.classes)))
        for index, cls in enumerate(self.classes):
            result[:, index] = self.models[cls].predict_proba(data_test)[:, 1]
        return result
