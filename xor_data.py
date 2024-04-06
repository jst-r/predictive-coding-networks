import torch as t
import torch.nn.functional as F
from matplotlib import pyplot as plt

def make_xor_data(device):
    X = t.rand(1000, 2, device=device) * 2 - 1
    xor_value = (X[:, 0] > 0) ^ (X[:, 1] > 0)
    labels = F.one_hot(xor_value.long(), 2).float()
    return X, labels

def plot_xor(X, labels):
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=labels[:, 0].cpu(), alpha=0.4)
