import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, model_name, history):
        self.history = history
        self.model_name = model_name

    def history_plot(self):
        plt.figure(figsize=(8, 10))
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss - ' + self.model_name, fontsize=12)
        plt.plot(self.history.history['loss'], color='blue', label='train')
        plt.plot(self.history.history['val_loss'], color='orange', label='val')
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(loc='upper right')

        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy ' + self.model_name, fontsize=10)
        plt.plot(self.history.history['accuracy'], color='blue', label='train')
        plt.plot(self.history.history['val_accuracy'], color='orange', label='val')
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend(loc='lower right')

        plt.show()