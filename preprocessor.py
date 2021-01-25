from utils import remove_punctuation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class Preprocessor:
    def __init__(self, config, classes):
        self.config = config
        self.orig_data = pd.read_csv(config['input_data'])
        self.test_data = pd.read_csv(config['test_data'])
        self.classes = classes
        self.prepared_data, self.prepared_labels = None, None
        self.test_id = None

    def process_training_data(self):
        print("Processing training data...")
        data = self.orig_data[self.config['input_text']]
        data.fillna("NA", inplace=True)
        self.prepared_data = remove_punctuation(data)
        self.prepared_labels = self.orig_data[self.classes]
        train_data, val_data, train_labels, val_labels = train_test_split(self.prepared_data,
                                                                          self.prepared_labels,
                                                                          test_size=0.2,
                                                                          random_state=0)

        train_x, val_x, test_x = self.vectorize(train_data, val_data)

        train_y = train_labels
        val_y = val_labels

        return train_x, val_x, train_y, val_y, test_x

    def vectorize(self, train_data, val_data):
        test_data = self.processing_test_data()
        print("Vectorizing data...")
        vectorizer = CountVectorizer()
        train_x = vectorizer.fit_transform(train_data)
        val_x = vectorizer.transform(val_data)
        test_x = vectorizer.transform(test_data)

        return train_x, val_x, test_x

    def processing_test_data(self):
        print("Processing test data...")
        self.test_id = self.test_data[self.config['test_id']]
        test_data = self.test_data[self.config['input_text']]
        test_data.fillna("NA", inplace=True)
        return remove_punctuation(test_data)




