# %%
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
from tqdm import tqdm


from src.prospective_configuration import PCLayer, PCModel
from src.utils import DEVICE, make_mnist_loaders


# %%

train_loader, val_loader = make_mnist_loaders(32)

# %%
def make_model():
    return PCModel(
        nn.Sequential(
            nn.Flatten(-3, -1),
            nn.Linear(28 * 28, 64, bias=False),
            PCLayer(),
            nn.LeakyReLU(),
            nn.Linear(64, 64, bias=False),
            PCLayer(),
            nn.LeakyReLU(),
            nn.Linear(64, 10, bias=False),
            nn.Softmax(),
        ),
        SummaryWriter()
    )


model = make_model().to(DEVICE)

RELAX_STEPS = 8
WEIGHT_STEPS = 4
WEIGHT_LR = 0.01

opt_weights = t.optim.AdamW(model.chunks.parameters(), lr=WEIGHT_LR / RELAX_STEPS / WEIGHT_STEPS)
loss_fn = nn.CrossEntropyLoss()

energies = []
losses = []

for epoch in tqdm(range(4)):
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.populate_activations(x)
        opt_energy = t.optim.SGD(model.pc_layers.parameters(), lr=1)
        opt_weights.zero_grad()
        for i in range(RELAX_STEPS):
            opt_energy.zero_grad()
            E = model.global_energy(x, loss_fn, y)
            E.backward()
            opt_energy.step()

        # for i in range(4):
        #     # opt_weights.zero_grad()
        #     E = model.global_energy(X, loss_fn, labels)
        #     E.backward()
        #     opt_weights.step()

        for _ in range(WEIGHT_STEPS):
            opt_weights.step()
    
        energies.append(E.item())
        losses.append(loss_fn(model.forward(x), y).item())
        print(f"loss {losses[-1]}\tE {energies[-1] - losses[-1]}")


plt.show()
plt.plot(energies)
plt.plot(losses)
plt.yscale("log")
plt.show()
print(f"energy: {E:.4}\tloss:{loss_fn(model.forward(x), y):.4}")

# %%
