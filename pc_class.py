# %%
import torch as t
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# %%
DEVICE = t.device("cpu")
# %%
X = t.rand(1000, 2, device=DEVICE) * 2 - 1
xor_value = (X[:, 0] > 0) ^ (X[:, 1] > 0)
labels = F.one_hot(xor_value.long(), 2).float()


# %%
def plot_xor(X, labels):
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=labels[:, 0].cpu(), alpha=0.4)


plot_xor(X, labels)


# %%
def make_model():
    return PCModel(
        nn.Sequential(
            nn.Linear(2, 8, bias=False),
            PCLayer(),
            nn.ReLU(),
            nn.Linear(8, 8, bias=False),
            PCLayer(),
            nn.ReLU(),
            nn.Linear(8, 2, bias=False),
            nn.Sigmoid(),
        )
    )


model = make_model()

# %% It can be trained with plain Adam
loss_fn = nn.BCELoss()
opt = t.optim.Adam(model.parameters(), lr=0.005)

# %%
for e in tqdm(range(500)):
    opt.zero_grad()
    y = model(X)
    loss = loss_fn(y, labels)
    loss.backward()
    opt.step()

    if e % 100 == 0:
        print(f"epoch: {e}\n\tloss: {loss}")

print(f"END\n\tloss: {loss}")

# %%
plot_xor(X, model(X).detach())


# %%
class PCLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation: t.Tensor | None = None
        self.activation_caching = False

    def forward(self, x):
        if self.activation_caching:
            self.activation = nn.Parameter(x)
        return x


class PCModel(nn.Module):
    def __init__(self, model: nn.Sequential):
        super().__init__()
        self.inner = model
        self.pc_layers = [
            module for module in self.inner.children() if isinstance(module, PCLayer)
        ]
        self.step = 0

    def forward(self, x: t.Tensor):
        return self.inner(x)

    def set_activation_caching(self, value: bool):
        for module in self.pc_layers:
            module.activation_caching = True

    def populate_activations(self, x):
        self.set_activation_caching(True)
        x = self.forward(x)
        self.set_activation_caching(False)
        return x

    def global_energy(
        self, x: t.Tensor, loss_fn, labels: t.Tensor
    ) -> tuple[t.Tensor, t.Tensor]:
        energy = t.tensor(0.0, requires_grad=True)
        for i, module in enumerate(self.inner.children()):
            if isinstance(module, PCLayer):
                error = module.activation - x
                dE = error.pow(2).sum() * 0.5
                writer.add_scalar(f"dE/{i}", dE.item(), self.step)
                energy = energy + dE
                x = module.activation
            else:
                x = module(x)

        loss = loss_fn(x, labels)

        writer.add_scalar("energy", E.item(), self.step)
        writer.add_scalar("loss", loss.item(), self.step)

        return energy, loss


# %%
WEIGHT_LR = 0.1
E_LR = 0.1

loss_fn = nn.BCELoss()
opt_weights = t.optim.SGD(model.parameters(), lr=WEIGHT_LR)

# %%
for e in tqdm(range(1000)):
    model.populate_activations(X)
    opt_energy = t.optim.SGD(map(lambda x: x.activation, model.pc_layers), lr=E_LR)

    for i in range(8):
        opt_weights.zero_grad()
        opt_energy.zero_grad()
        E, loss = model.global_energy(X, loss_fn, labels)
        (loss).backward()
        opt_energy.step()
        opt_weights.step()

    if e % 50 == 0:
        print(f"epoch: {e}\n\tglobal energy: {E}\n\tloss: {loss}")

# %%
plot_xor(X, model(X).detach())

# %%

writer = SummaryWriter()

WEIGHT_LR = 0.01
E_LR = 0.01


model = make_model()
model.populate_activations(X)

opt_weights = t.optim.SGD(model.parameters(), lr=WEIGHT_LR)
opt_energy = t.optim.SGD(map(lambda x: x.activation, model.pc_layers), lr=E_LR)

energies = []
losses = []

try:
    for i in range(64):
        model.step = i
        opt_weights.zero_grad()
        opt_energy.zero_grad()
        E, loss = model.global_energy(X, loss_fn, labels)
        (E + loss).backward()
        opt_energy.step()
        opt_weights.step()
finally:
    writer.close()
# %%

plt.plot(energies, label="energy")
plt.plot(losses, label="loss")
plt.legend()
plt.show()
# %%
# %%
