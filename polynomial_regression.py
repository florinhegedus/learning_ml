import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import save_image_at_epoch, create_video


class PolynomialRegressionModel(nn.Module):
    def __init__(self, degree):
        super(PolynomialRegressionModel, self).__init__()
        self.degree = degree
        self.linear = nn.Linear(self.degree, 1, bias=True)

    def forward(self, x):
        x = torch.stack([x**i for i in range(self.degree, 0, -1)], dim=1)
        out = self.linear(x)
        return out


def generate_data(num_examples: int): 
    # X = torch.linspace(-4, 7, num_examples)
    # y = X**3 - 4*X**2 + X + 5
    X = torch.linspace(-4, 4, num_examples)
    y = X**2 - 3*X + 2
    y += torch.normal(0, 0.2, y.shape)
    return X, y.view(-1, 1)


def get_dataloader(data_arrays, batch_size, shuffle):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int, visu: bool=True):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.000001, momentum=0.99)

    for epoch in range(epochs):
        # Train loop
        model.train()
        for X, y in train_dataloader:
            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                val_preds = model(X_val)
                val_loss += criterion(val_preds, y_val).item()

        val_loss /= len(val_dataloader)
        print(f'epoch {epoch}: validation loss={val_loss}')

        if visu:
            save_image_at_epoch(model, X_val, y_val, epoch)


if __name__ == '__main__':
    batch_size, epochs = 32, 100
    num_train_examples = 512
    num_val_examples = batch_size
    polynom_degree = 2
    visu = True
    # Training data
    X_train, y_train = generate_data(num_train_examples)
    train_dataloader = get_dataloader((X_train, y_train), batch_size, True)

    # Validation data
    X_val, y_val = generate_data(num_val_examples)
    val_dataloader = get_dataloader((X_val, y_val), batch_size, False)

    model = PolynomialRegressionModel(degree=polynom_degree)
    train(model, train_dataloader, val_dataloader, epochs=epochs, visu=visu)

    print(f"predicted alpha = {model.linear.weight.data}")
    print(f"predicted beta = {model.linear.bias.data.item()}")

    if visu:
        create_video(epochs, fps=30)