# %%
import torch as t
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

# %%
DEVICE = t.device("cuda")
# %%
noisy = t.randn(1000, 2)
xor_value = (noisy[:, 0] > 0) ^ (noisy[:, 1] > 0)
labels = F.one_hot(xor_value.long(), 2).float()

# %%
plt.scatter(noisy[:, 0], noisy[:, 1], c=labels[:, 0], alpha=0.4)
# %%
model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2), nn.Softmax())
# %%
loss_fn = nn.BCELoss()
opt = t.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5)
# %%
for e in range(1, 1001):
    loss = loss_fn(model(noisy), labels)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if e % 100 == 0:
        print(f"Epoch: {e}; loss: {loss}")
# %%
plt.scatter(noisy[:, 0], noisy[:, 1], c=model(noisy).detach()[:, 0], alpha=0.4)

# %%
list(model.parameters())


# %%
class PCLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.activations = nn.Parameter(t.zeros(dim))

    def forward(self, input_):
        return input_


class PCOptimizer:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 0.05
        pc_layers = list(filter(lambda m: isinstance(m, PCLayer), self.model.modules()))
        self.optimizer = t.optim.SGD(
            map(lambda m: m.parameters(), pc_layers), lr=self.lr
        )
