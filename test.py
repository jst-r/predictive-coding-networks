# %%
import torch as t
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from prospective_configuration import PCLayer, PCModel
from xor_data import make_xor_data, plot_xor

DEVICE = t.device("cpu")
X, labels = make_xor_data(DEVICE)

# %%
def make_model():
    return PCModel(
        nn.Sequential(
            nn.Linear(2, 8, bias=False),
            PCLayer(),
            # nn.ReLU(),
            # nn.Linear(8, 8, bias=False),
            # PCLayer(),
            nn.ReLU(),
            nn.Linear(8, 2, bias=False),
            nn.Sigmoid(),
        ),
        SummaryWriter()
    )


model = make_model()
model.populate_activations(X)

opt_weights = t.optim.Adam(model.chunks.parameters(), lr=0.5)
opt_energy = t.optim.SGD(model.pc_layers.parameters(), lr=0.5, momentum=0.9)
loss_fn = nn.BCELoss()

for epoch in range(500):
    for i in range(4):
        opt_energy.zero_grad()
        E = model._global_energy(X, loss_fn, labels)
        E.backward()
        opt_energy.step()

    for i in range(4):
        opt_weights.zero_grad()
        E = model._global_energy(X, loss_fn, labels)
        E.backward()
        opt_weights.step()
    print(f"{epoch} energy: {E:.4}\tloss:{loss_fn(model.forward(X), labels):.4}")

plot_xor(X, model.forward(X).detach())
# %%
plot_xor(X, labels)
# %% It can be trained with plain Adam
model = make_model()
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

plot_xor(X, model(X).detach())

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
