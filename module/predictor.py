import csv


class Predictor:
    def __init__(self, config, model, logger):
        self.logger = logger
        self.config = config
        self.model = model

    def predict(self, test_data):
        return self.model.predict(test_data)

    def predict_prob(self, test_data):
        return self.model.predict_prob(test_data)

    def save_to_csv(self, id, probs):
        header = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.logger.info("Saving prediction result to a csv file...")
        with open(self.config['output_path'], 'w') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(header)
            for ids, prob in zip(id, probs.tolist()):
                writer.writerow([ids] + prob)
        self.logger.info("Done. Prediction completed!")