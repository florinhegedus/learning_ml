import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs


class SVM(nn.Module):
    def __init__(self, input_dim, C=1.0):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.C = C

    def forward(self, x):
        return self.fc(x)

    def fit(self, X, y, epochs=100, lr=0.00001):
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = self.hinge_loss(outputs, y)
            if epoch % 100 == 0:
                print(f"{epoch}: loss={loss}")
            loss.backward()
            optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            outputs = self(X)
        return torch.sign(outputs)

    def hinge_loss(self, outputs, y):
        return torch.mean(torch.clamp(1 - outputs.t() * y, min=0)) + self.C * torch.norm(self.fc.weight)**2
    

def get_data():
    # Generate data
    X, y = make_blobs(n_samples=100, centers=2, random_state=6)

    # Convert labels from {0, 1} to {-1, 1} for SVM
    y = 2*y - 1

    # Convert data to tensors for PyTorch SVM
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).view(-1, 1)

    return X, y


def accuracy(predictions, true_labels):
    correct = torch.sum(predictions == true_labels)
    total = len(true_labels)
    return correct / total


if __name__ == '__main__':
    X, y = get_data()
    svm = SVM(input_dim=2)
    svm.fit(X, y, epochs=10000)
    predictions = svm.predict(X)
    acc = accuracy(predictions, y)

    print(f"Accuracy: {acc * 100:.2f}%")
