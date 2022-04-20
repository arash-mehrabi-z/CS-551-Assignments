import numpy as np
CHROMOSOME_COUNTER = 0

class Chromosome():
    def __init__(self, weight):
        self.weight = weight

        global CHROMOSOME_COUNTER
        CHROMOSOME_COUNTER += 1
        self.identifier = CHROMOSOME_COUNTER.__str__()

    def get_weights(self):
        w1 = self.weight[:6].reshape((2, 3))
        w2 = self.weight[6:].reshape((3, 2))
        return w1, w2

    def fitness(self, y_pred, y_truth):
        N = y_truth.shape[0]
        squarred_error = np.power(y_truth - y_pred, 2)
        mse = np.sum(squarred_error) / N
        self.mse = mse
        return mse

    def predict(self, X):
        w1, w2 = self.get_weights()
        H = np.dot(w1, X)
        F = np.dot(w2, H)
        y_pred = np.sum(F, axis=0)
        return y_pred

    def __repr__(self):
        return self.identifier
