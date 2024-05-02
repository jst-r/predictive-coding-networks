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
def make_model(writer: SummaryWriter | None = None):
    return PCModel(
        nn.Sequential(
            nn.Flatten(-3, -1),
            nn.Linear(28 * 28, 64, bias=False),
            PCLayer(),
            nn.LeakyReLU(),
            nn.Linear(64, 64, bias=False),
            PCLayer(),
            nn.LeakyReLU(),
            nn.Linear(64, 64, bias=False),
            PCLayer(),
            nn.LeakyReLU(),
            nn.Linear(64, 10, bias=False),
            nn.Softmax(),
        ),
        writer if writer else SummaryWriter()
    )

RELAX_STEPS = 8
WEIGHT_STEPS = 1
WEIGHT_LR = 0.05

def train_step(
    model: PCModel,
    x: t.Tensor,
    y: t.Tensor,
    loss_fn: nn.Module,
    opt_weights: t.optim.Optimizer,
    opt_energy: t.optim.Optimizer
    ):
    x, y = x.to(DEVICE), y.to(DEVICE)
    model.populate_activations(x)
    opt_energy = t.optim.SGD(model.pc_layers.parameters(), lr=1)
    opt_weights.zero_grad()
    for i in range(RELAX_STEPS):
        opt_energy.zero_grad()
        [E, loss] = model.global_energy(x, loss_fn, y)
        if i == 0:
            model.writer.add_scalar("loss/before_relaxation", loss.item(), model.step)
            model.writer.add_scalar("energy_before_relaxation", E.item(), model.step)
        if i == RELAX_STEPS - 1:
            model.writer.add_scalar("loss/after_relaxation", loss.item(), model.step)
            model.writer.add_scalar("energy_after_relaxation", E.item(), model.step)
        (E + loss).backward()
        opt_energy.step()

    # for i in range(4):
    #     # opt_weights.zero_grad()
    #     E = model.global_energy(X, loss_fn, labels)
    #     E.backward()
    #     opt_weights.step()

    for _ in range(WEIGHT_STEPS):
        opt_weights.step()


    return E

# %%
model = make_model().to(DEVICE)

opt_weights = t.optim.AdamW(model.chunks.parameters(), lr=WEIGHT_LR / RELAX_STEPS / WEIGHT_STEPS)
loss_fn = nn.CrossEntropyLoss()


for epoch in tqdm(range(1)):
    for i, [x, y] in enumerate(train_loader):
        model.step = i
        train_step(model, x, y, loss_fn, opt_weights, opt_weights)



# %%
