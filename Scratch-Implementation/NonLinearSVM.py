import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class NonLSVM:
    def __init__(self, kernel_type='poly', degree=2, sigma_val=0.1, max_epochs=100, learning_rate=0.001):
        self.alphas = None
        self.bias = 0
        self.degree = degree
        self.C = 1
        self.sigma = sigma_val
        self.epochs = max_epochs
        self.learning_rate = learning_rate

        if kernel_type == 'poly':
            self.kernel_func = self.polynomial_kernel
        elif kernel_type == 'rbf':
            self.kernel_func = self.gaussian_kernel
        else:
            raise Exception("Kernel type must be 'poly' or 'rbf'")

    def polynomial_kernel(self, X, Z):
        return (self.C + X.dot(Z.T)) ** self.degree

    def gaussian_kernel(self, X, Z):
        return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(X[:, np.newaxis] - Z[np.newaxis, :], axis=2) ** 2)

    def train(self, X, y):
        self.X = X
        self.y = y
        self.alphas = np.random.random(X.shape[0])
        self.bias = 0
        ones = np.ones(X.shape[0])

        y_kernel = np.outer(y, y) * self.kernel_func(X, X)

        for epoch in range(self.epochs):
            gradient = ones - y_kernel.dot(self.alphas)
            self.alphas += self.learning_rate * gradient
            self.alphas[self.alphas > self.C] = self.C
            self.alphas[self.alphas < 0] = 0

        alpha_indices = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
        b_values = []

        for index in alpha_indices:
            b_values.append(y[index] - (self.alphas * y).dot(self.kernel_func(X, X[index])))

        self.bias = np.mean(b_values)

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def get_accuracy(self, true_labels, predicted_labels):
        return accuracy_score(true_labels, predicted_labels)

    def decision_function(self, X):
        return (self.alphas * self.y).dot(self.kernel_func(self.X, X)) + self.bias

    def plot_decision_boundary(self, X, y, title):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', alpha=0.5)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 50)
        yy = np.linspace(ylim[0], ylim[1], 50)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'])
        plt.title(title)
        plt.show()

