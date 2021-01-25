import csv


class Predictor:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def predict(self, test_data):
        return self.model.predict(test_data)

    def predict_prob(self, test_data):
        return self.model.predict_prob(test_data)

    def save_to_csv(self, id, probs):
        header = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        print("Saving result to csv file......")
        print(self.config['output_path'])
        with open(self.config['output_path'], 'w') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(header)
            for ids, prob in zip(id, probs.tolist()):
                writer.writerow([ids] + prob)