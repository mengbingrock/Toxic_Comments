from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Dropout, Flatten, MaxPooling1D

class TextCNN:
    def __init__(self, config, classes, vocab_size, logger):
        self.models = {}
        self.logger = logger
        self.vocab_size = vocab_size
        self.config = config
        self.classes = classes
        self.n_of_classes = len(classes)
        self.model = self._build()

    def _show_training_config_para(self):
        title = "Training config parameters"
        self.logger.info(title.center(40, '-'))
        self.logger.info("---model_name = {}".format(self.config['model_name']))
        self.logger.info("---max_input_len = {}".format(self.config['max_len']))
        self.logger.info("---batch_size = {}".format(self.config['batch_size']))
        self.logger.info("---dropout = {}".format(self.config['dropout']))
        self.logger.info("---epochs = {}".format(self.config['epochs']))

    def _build(self):
        self._show_training_config_para()
        model = Sequential()
        model.add(Embedding(self.vocab_size,
                            self.config['embedding_col'],
                            embeddings_initializer='uniform',
                            input_length=self.config['max_len'],
                            trainable=True))
        model.add(Conv1D(128, 7, activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(256, 5, activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(512, 3, activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.config['dropout']))
        model.add(Dense(self.n_of_classes, activation=None))
        model.add(Dense(self.n_of_classes, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        return model

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        fitted = self.model.fit(train_x,
                                train_y,
                                epochs=self.config['epochs'],
                                verbose=True,
                                validation_data=(validate_x, validate_y),
                                batch_size=self.config['batch_size'])

        predictions = self.predict(validate_x)

        return predictions, fitted

    def predict(self, validate_x):
        probs = self.model.predict(validate_x)
        return probs >= 0.5

    def predict_prob(self, validate_x):
        return self.model.predict(validate_x)