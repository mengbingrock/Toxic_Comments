import yaml
import argparse
from preprocessor import Preprocessor
from naive_bayes import NaiveBayes
from metrics import Metrics
from predictor import Predictor


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Processing command line')
    parser.add_argument('--config', type=str, required=True)
    config_args = parser.parse_args()
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    with open(config_args.config) as cfg:
        try:
            config = yaml.safe_load(cfg)
            preprocessor = Preprocessor(config['preprocessing'], classes)
            data_train, data_val, labels_train, labels_val, data_test = preprocessor.process_training_data()

            model = NaiveBayes(classes)
            model.fit(data_train, labels_train)

            predictor = Predictor(config['predict'], model)
            predictions = predictor.predict(data_val)

            metrics = Metrics(predictions, labels_val)
            metrics.show_accuracy()

            # predict probability test dataset
            probs = predictor.predict_prob(data_test)
            predictor.save_to_csv(preprocessor.test_id, probs)

        except yaml.YAMLError as err:
            print("config file error : {}".format(err))