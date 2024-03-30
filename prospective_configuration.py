# %%
import torch as t
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

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
def forward(x):
    return F.softmax(l1(F.relu(l0(x))), dim=-1)


def prepare_activations(x, y):
    actIn = nn.Parameter(x)
    act0 = nn.Parameter(l0(actIn))
    actOut = y

    return [actIn, act0, actOut]


def global_energy(activations):
    actIn, act0, actOut = activations

    def vec_square(x):
        return t.einsum("bi,bi->b", x, x)

    E0 = vec_square(act0 - l0(actIn)).mean()
    E1 = vec_square(actOut - F.softmax(l1(F.relu(act0)), dim=-1)).mean()

    return E0 + E1


def relax_step(activations: list[nn.Parameter]):
    for a in activations:
        a.grad = None
    E = global_energy(activations)
    E.backward()
    with t.no_grad():
        activations[1] -= activations[1].grad


# %%
l0 = nn.Linear(2, 4, device=DEVICE)
l1 = nn.Linear(4, 2, device=DEVICE)
# %%
loss_fn = nn.BCELoss()
opt_weights = t.optim.AdamW([l0.weight, l0.bias, l1.weight, l1.bias], lr=0.01)
# %%
for e in tqdm(range(500)):
    acts = prepare_activations(X, labels)
    opt_energy = t.optim.AdamW(acts[1:-1], lr=0.05)

    for i in range(16):
        opt_weights.zero_grad()
        opt_energy.zero_grad()
        E = global_energy(acts)
        E.backward()
        opt_energy.step()
        opt_weights.step()

    if e % 100 == 0:
        print(
            f"epoch: {e}\n\tglobal energy: {E}\n\tloss: {loss_fn(forward(X), labels)}"
        )

print(f"END\n\tglobal energy: {E}\n\tloss: {loss_fn(forward(X), labels)}")

# %%
plot_xor(X, forward(X).detach())

# %%
acts = prepare_activations(X, labels)
acts[-1] = F.softmax(l1(F.relu(acts[1])), dim=-1).detach()
opt_energy = t.optim.SGD(acts[1:], lr=1)
for _ in range(16):
    opt_energy.zero_grad()
    E = global_energy(acts)
    E.backward()
    opt_energy.step()
    print(E)

plot_xor(X, acts[-1].detach())
# %%
acts[-1]

# %%
l1.weight
