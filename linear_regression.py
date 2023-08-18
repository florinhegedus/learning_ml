import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import save_image_at_epoch, create_video


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


def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int, visu: bool=True):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

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
    alpha, beta = 3, 0.5
    batch_size, epochs = 16, 20
    num_train_examples = 256
    visu = True
    # Training data
    X_train, y_train = generate_data(alpha, beta, num_train_examples)
    train_dataloader = get_dataloader((X_train, y_train), batch_size, True)

    # Validation data
    X_val, y_val = generate_data(alpha, beta, batch_size)
    val_dataloader = get_dataloader((X_val, y_val), batch_size, False)

    model = LinearRegressionModel()
    train(model, train_dataloader, val_dataloader, epochs=epochs, visu=visu)

    print(f"predicted alpha = {model.linear.weight.data.item()}")
    print(f"predicted beta = {model.linear.bias.data.item()}")

    if visu:
        create_video(epochs, fps=3)