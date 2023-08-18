import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        out = self.linear(x)
        return out


def generate_data(alpha: int, beta: int, num_examples: int): 
    X = torch.linspace(0.1, 10, num_examples)
    y = alpha * X + beta
    y += torch.normal(0, 1.0, y.shape)
    return X.view(-1, 1), y.view(-1, 1)


def get_dataloader(data_arrays, batch_size, shuffle):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(model: nn.Module, dataloader: DataLoader, epochs: int):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        for X, y in dataloader:
            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch}: loss={loss}')


if __name__ == '__main__':
    X, y = generate_data(5, -3, 1024)
    dataloader = get_dataloader((X, y), 16, True)
    model = LinearRegressionModel()
    train(model, dataloader, epochs=1000)

    print(f"predicted alpha = {model.linear.weight.data.item()}")
    print(f"predicted beta = {model.linear.bias.data.item()}")