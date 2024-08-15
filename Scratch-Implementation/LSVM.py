import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class LinearSVM:
    def __init__(self, margin_type="hard", learning_rate=0.001, tradeoff=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.tradeoff = tradeoff
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

        if margin_type == "hard":
            self.tradeoff = 1

    def train(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = self.X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for i, data_point in enumerate(self.X):
                if self.y[i] * (np.dot(data_point, self.weights) - self.bias) >= 1:
                    self.weights -= self.learning_rate * (2 * self.tradeoff * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.tradeoff * self.weights - np.dot(data_point, self.y[i]))
                    self.bias -= self.learning_rate * self.y[i]

    def predict(self, X):
        predictions = np.dot(X, self.weights) - self.bias
        y_pred = [1 if val > 0 else -1 for val in predictions]
        return y_pred

    def compute_accuracy(self, y_pred):
        return accuracy_score(y_pred, self.y)

    def compute_hyperplane(self, x, offset):
        return (-self.weights[0] * x + self.bias + offset) / self.weights[1]

    def visualize_decision_boundary(self, plot_title="SVM Decision Boundary"):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='o', c=self.y)

        x0_1 = np.amin(self.X[:, 0])
        x0_2 = np.amax(self.X[:, 0])

        x1_1 = self.compute_hyperplane(x0_1, 0)
        x1_2 = self.compute_hyperplane(x0_2, 0)

        x1_1_m = self.compute_hyperplane(x0_1, -1)
        x1_2_m = self.compute_hyperplane(x0_2, -1)

        x1_1_p = self.compute_hyperplane(x0_1, 1)
        x1_2_p = self.compute_hyperplane(x0_2, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

        x1_min = np.amin(self.X[:, 1])
        x1_max = np.amax(self.X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.title(plot_title)
        plt.show()
