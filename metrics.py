from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class Metrics:
    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels

    def show_accuracy(self):
        accuracy = accuracy_score(self.labels, self.predictions)
        report = classification_report(self.labels, self.predictions)
        print("==========Accuracy score============\n")
        print(accuracy)
        print("==========Classification report============\n")
        print(report)
