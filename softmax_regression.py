import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import visualize_points, create_video


class SoftmaxRegressionModel(nn.Module):
    def __init__(self, num_classes):
        super(SoftmaxRegressionModel, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.linear(x)
        probs = self.softmax(logits)
        return logits, probs
    

def generate_point_cloud(center_x: float, center_y: float, std: float, num_points: int):
    x = torch.normal(center_x, std, size=(num_points, ))
    y = torch.normal(center_y, std, size=(num_points, ))
    return torch.stack((x, y), dim=1)
    

def generate_data(num_samples: int):
    '''
        Generate 4 point clouds (the 4 classes for the softmax regression)
    '''
    num_classes = 4
    num_points_per_class = int(num_samples / num_classes)

    pc1 = generate_point_cloud(0, 0, 0.1, num_points_per_class)
    pc2 = generate_point_cloud(4, 0, 0.1, num_points_per_class)
    pc3 = generate_point_cloud(0, 4, 0.1, num_points_per_class)
    pc4 = generate_point_cloud(4, 4, 0.1, num_points_per_class)
    X = torch.cat((pc1, pc2, pc3, pc4))

    y = torch.cat([torch.tensor([i] * num_points_per_class) for i in range(num_classes)])
    y = F.one_hot(y, num_classes=num_classes).float()

    return X, y


def get_dataloader(data_tensors, batch_size, shuffle):
    dataset = TensorDataset(*data_tensors)
    return DataLoader(dataset, batch_size, shuffle)


def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int, visu: bool):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00005, momentum=0.99)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for X, y in iter(train_dataloader):
            logits, probs = model(X)
            loss = criterion(y, logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                logits, probs = model(X_val)
                val_loss += criterion(logits, y_val).item()

                # Getting the predicted class labels
                predictions = torch.argmax(probs, dim=1)
                y_val = torch.argmax(y_val, dim=1)
                correct += (predictions == y_val).sum().item()

                if visu:
                    visualize_points(X_val, predictions, num_classes=4, epoch=epoch)

        val_loss /= len(val_dataloader)
        accuracy = correct / len(val_dataloader.dataset)
        print(f'epoch {epoch}: validation loss={val_loss}, accuracy: {accuracy}')


if __name__ == '__main__':
    num_samples = 32
    batch_size = 16
    epochs = 250
    num_classes = 4
    visu = True

    X_train, y_train = generate_data(num_samples)
    train_dataloader = get_dataloader((X_train, y_train), batch_size, shuffle=True)

    X_val, y_val = generate_data(batch_size)
    val_dataloader = get_dataloader((X_val, y_val), batch_size, shuffle=False)

    model = SoftmaxRegressionModel(num_classes)

    train(model, train_dataloader, val_dataloader, epochs, visu)

    if visu:
        create_video(epochs, fps=30)
