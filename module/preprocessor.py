import pandas as pd
import numpy as np
import io
from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils import remove_punctuation


class Preprocessor:
    def __init__(self, config, classes, logger):
        self.config = config
        self.logger = logger
        self.classes = classes
        self.train_data, self.train_labels = None, None
        self.train_x, self.validate_x, self.train_y, self.validate_y = None, None, None, None
        self.embedding_matrix = None
        self.vocab_size = 0
        self._load_data()
        self.word2ind = {}
        self.ind2word = {}

    def _show_config_parameters(self):
        title = "config parameters"
        self.logger.info("Getting config parameters...")
        self.logger.info(title.center(40, '-'))
        self.logger.info("---split_ratio = {}".format(self.config['split_ratio']))
        self.logger.info("---random_state = {}".format(self.config['random_state']))
        self.logger.info("---max_len = {}".format(self.config['max_len']))

    def _load_data(self):
        self._show_config_parameters()
        self.logger.info("Loading training and test data...")
        orig_data = pd.read_csv(self.config['input_data_path'])
        test_data = pd.read_csv(self.config['test_data_path'])
        orig_data[self.config['input_text']].fillna("unknown", inplace=True)
        test_data[self.config['input_text']].fillna("unknown", inplace=True)

        self.train_data, self.train_labels = self._parse_trained_data(orig_data)
        self.logger.info("Spliting datasets, the split ratio is {}, random state is {}".format(self.config['split_ratio'],
                                                                                               self.config['random_state']))
        self.train_x, self.validate_x, self.train_y, self.validate_y = train_test_split(
                                                                       self.train_data,
                                                                       self.train_labels,
                                                                       test_size=self.config['split_ratio'],
                                                                       random_state=self.config['random_state'])

        self.test_data, self.test_id = self._parse_test_data(test_data)
        self.logger.info("Loading done.")

    def _parse_trained_data(self, orig_data):
        self.logger.info("Parsing input text...")
        text = orig_data[self.config['input_text']]
        labels = orig_data[self.classes].values
        train_data = remove_punctuation(text).values
        return train_data, labels

    def _parse_test_data(self, test_data):
        self.logger.info("Parsing test dataset...")
        text = remove_punctuation(test_data[self.config['input_text']]).values
        ids = test_data.id.values
        return text, ids

    def process(self):
        text_converter = self.config['text_converter']

        train_data, train_labels, train_x, validate_x, train_y, validate_y, test_data = \
        self.train_data, self.train_labels, self.train_x, self.validate_x, self.train_y, self.validate_y, self.test_data

        if text_converter == 'neural_network_vecterization':
            train_x, validate_x, test_data = self.nn_vecterization(train_x, validate_x, test_data)

        return train_data, train_labels, train_x, validate_x, train_y, validate_y, test_data

    def nn_vecterization(self, train_x, validate_x, test_data):
        # initialize hash table for word-->id and id-->word
        self.logger.info("Vecterizing data for neural network training...")
        specialchars = ['<pad>', '<unk>']

        pretrained_embedding = self.config.get('pretrained_embedding', None)

        if pretrained_embedding is not None:
            self.logger.info("Loading pretrained embeddings {}".format(pretrained_embedding))
            embedding = Preprocessor.load_word_embedding(pretrained_embedding)
            self.logger.info("Loading done")

            self.logger.info("Creating vocabulary...")
            vocabs = specialchars + list(embedding.keys())
            self.vocab_size = len(vocabs)
            self.embedding_matrix = np.zeros((self.vocab_size, self.config['embedding_col']))

            for token in specialchars:
                embedding[token] = np.random.uniform(low=-1, high=1, size=(self.config['embedding_col']))

            for index, word in enumerate(vocabs):
                self.word2ind[word] = index
                self.ind2word[index] = word
                self.embedding_matrix[index] = embedding[word]
        else:
            def add_word(word2ind, ind2word, word):
                if word in word2ind:
                    return
                ind2word[len(word2ind)] = word
                word2ind[word] = len(word2ind)

            for char in specialchars:
                add_word(self.word2ind, self.ind2word, char)

            self.logger.info("Creating vocabulary....")
            for sentence in train_x:
                for word in sentence:
                    add_word(self.word2ind, self.ind2word, word)
            self.vocab_size = len(self.word2ind.keys())

        self.logger.info("Done. Got {} words".format(len(self.word2ind.keys())))
        self.logger.info("Preparing data for training...")

        train_x_in = []
        for sentence in train_x:
            indices = [self.word2ind.get(word, self.word2ind['<unk>']) for word in sentence]
            train_x_in.append(indices)

        train_x_in = np.array(train_x_in)

        validate_x_in = []
        for sentence in validate_x:
            indices = [self.word2ind.get(word, self.word2ind['<unk>']) for word in sentence]
            validate_x_in.append(indices)

        validate_x_in = np.array(validate_x_in)

        test_data_in = []
        for sentence in test_data:
            indices = [self.word2ind.get(word, self.word2ind['<unk>']) for word in sentence]
            test_data_in.append(indices)

        test_data_in = np.array(test_data_in)

        train_x_in = keras.preprocessing.sequence.pad_sequences(train_x_in,
                                                                maxlen=self.config['max_len'],
                                                                padding='post',
                                                                value=self.word2ind['<pad>'])

        validate_x_in = keras.preprocessing.sequence.pad_sequences(validate_x_in,
                                                                   maxlen=self.config['max_len'],
                                                                   padding='post',
                                                                   value=self.word2ind['<pad>'])

        test_data_in = keras.preprocessing.sequence.pad_sequences(test_data_in,
                                                                  maxlen=self.config['max_len'],
                                                                  padding='post',
                                                                  value=self.word2ind['<pad>'])
        return train_x_in, validate_x_in, test_data_in

    @staticmethod
    def load_word_embedding(filename):
        file_in = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        data = {}

        for line in file_in:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(list(map(float, tokens[1:])))
        return data
