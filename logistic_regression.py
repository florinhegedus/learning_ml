import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import visualize_points


class LogisticRegressionModel(nn.Module):
    def __init__(self) -> None:
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out


def generate_point_cloud(center_x: float, center_y: float, std_dev: float, num_points: int):
    x = torch.normal(float(center_x), float(std_dev), size=(num_points,))
    y = torch.normal(float(center_y), float(std_dev), size=(num_points,))
    return torch.stack((x, y), dim=1)


def generate_data(num_samples: int):
    # Input of the model will be 2D coordinates: X = [(x, y), ...]
    num_points_per_class = int(num_samples/2)
    pc1 = generate_point_cloud(0, 4, 2, num_points_per_class)
    pc2 = generate_point_cloud(4, 0, 2, num_points_per_class)
    X = torch.cat((pc1, pc2))
    y = torch.cat((torch.zeros(num_points_per_class), torch.ones(num_points_per_class)))
    return X, y.view(-1, 1)


def get_dataloader(data_arrays, batch_size, shuffle):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.99)

    for epoch in range(epochs):
        model.train()
        for X, y in iter(train_dataloader):
            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                val_preds = model(X_val)
                val_loss += criterion(val_preds, y_val).item()
                y_pred_cls = (val_preds > 0.5).float()
                accuracy = (y_pred_cls == y_val).float().mean()

                visualize_points(X_val, y_pred_cls, num_classes=2, epoch=epoch)

        val_loss /= len(val_dataloader)
        print(f'epoch {epoch}: val loss={val_loss}, val accuracy: {accuracy.item()}')


if __name__ == '__main__':
    batch_size = 16
    num_train_examples = 512
    epochs = 10

    X_train, y_train = generate_data(num_train_examples)
    train_dataloader = get_dataloader((X_train, y_train), batch_size, shuffle=True)

    X_val, y_val = generate_data(batch_size)
    val_dataloader = get_dataloader((X_val, y_val), batch_size, shuffle=False)

    model = LogisticRegressionModel()
    train(model, train_dataloader, val_dataloader, epochs)

