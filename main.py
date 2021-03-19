import yaml
import argparse
import logging
from module.preprocessor import Preprocessor
from module.trainer import Trainer
from module.predictor import Predictor
from metrics.metrics import Metrics
import os


if __name__ == "__main__":

    # new changes
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    parser = argparse.ArgumentParser(description='Processing command line')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--loglevel', type=str, default="INFO")
    args = parser.parse_args()

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=args.loglevel)
    logger = logging.getLogger('global_logger')

    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    with open(args.config) as cfg:
        try:
            config = yaml.safe_load(cfg)
            preprocessor = Preprocessor(config['preprocessing'], classes, logger)
            train_data, train_labels, train_x, validate_x, train_y, validate_y, test_data = preprocessor.process()

            vocab_size = preprocessor.vocab_size
            pretrained_embedding = preprocessor.embedding_matrix
            trainer = Trainer(config['training'], classes, logger, vocab_size, pretrained_embedding)
            model, accuracy, cls_report, history = trainer.fit_and_validate(train_x, train_y, validate_x, validate_y)
            logger.info("Accuracy : {}".format(accuracy))
            logger.info("\n{}\n".format(cls_report))

            metric = Metrics(config['training']['model_name'], history)
            metric.history_plot()

            predictor = Predictor(config['predict'], model, logger)
            probs = predictor.predict_prob(test_data)
            predictor.save_to_csv(preprocessor.test_id, probs)

        except yaml.YAMLError as err:
            print("config file error : {}".format(err))
