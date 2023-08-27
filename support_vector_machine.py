import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


if __name__ == '__main__':
    # Generate linearly separable data
    X, y = make_blobs(n_samples=100, centers=2, random_state=6)

    # Convert labels from {0, 1} to {-1, 1} for SVM
    y = 2*y - 1

    # Train the SVM
    clf = LinearSVM()
    clf.fit(X, y)

    # Predict
    predictions = clf.predict(X)

    # Calculate accuracy
    accuracy = np.mean(y == predictions)
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Plotting the decision boundary
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
    ax = plt.gca()
    xlim = ax.get_xlim()
    xx = np.linspace(xlim[0], xlim[1])
    yy = -clf.w[0] / clf.w[1] * xx + (clf.b / clf.w[1])
    plt.plot(xx, yy, 'k-')
    plt.show()
