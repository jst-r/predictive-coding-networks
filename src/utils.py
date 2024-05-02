import torch as t
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms

DEVICE = t.device("cuda" if t.torch.cuda.is_available() else "cpu")

def make_xor_data(device):
    X = t.rand(1000, 2, device=device) * 2 - 1
    xor_value = (X[:, 0] > 0) ^ (X[:, 1] > 0)
    labels = F.one_hot(xor_value.long(), 2).float()
    return X, labels

def plot_xor(X, labels):
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=labels[:, 0].cpu(), alpha=0.4)

def make_mnist_loaders(batch_size: int):
    # TODO proper path
    dataset = torchvision.datasets.MNIST(root = '../data/', download = True, transform=transforms.ToTensor())

    train_data, validation_data = random_split(dataset, [50000, 10000])

    train_loader = DataLoader(train_data, batch_size, shuffle = True, pin_memory=True)
    val_loader = DataLoader(validation_data, batch_size, shuffle = False, pin_memory=True)

    return train_loader, val_loader