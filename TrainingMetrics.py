import numpy as np


class TrainingMetrics:
    def __init__(self, n_epochs):
        self.losses = np.zeros(n_epochs)
        self.precisions = np.zeros(n_epochs)
        self.recalls = np.zeros(n_epochs)
        self.f1s = np.zeros(n_epochs)

    def update(self, epoch, loss, precision, recall, f1):
        self.losses[epoch] = loss
        self.precisions[epoch] = precision
        self.recalls[epoch] = recall
        self.f1s[epoch] = f1

