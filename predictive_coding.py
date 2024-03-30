# %%
# ruff: noqa: F722 E402
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# %%
import torch as t
from torch.nn import functional as F
from torchvision import datasets, transforms

from einops import rearrange, reduce, repeat

from icecream import ic

# %%
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import List

typechecker = typechecked

# %%
from dataclasses import dataclass


def relu_gradient(x: t.Tensor):
    return t.where(x > 0, t.ones_like(x), t.zeros_like(x))


def sigmoid_gradient(x: t.Tensor):
    return t.sigmoid(x) * (1 - t.sigmoid(x))


@dataclass
class PCNConfig:
    learning_rate = 0.1
    integration_step = 0.01
    n_relaxation_steps = 8


class PCN:
    def __init__(self, cfg: PCNConfig, shape: List[int], device="cuda:0") -> None:
        self.cfg = cfg
        self.n_layers = len(shape)
        self.shape = shape
        assert (
            self.n_layers >= 2
        ), "at least two dims are required (input dimension and output dimension)"

        self.device = t.device(device)
        with self.device:
            in_dim, *hidded, out_dim = shape

            self.activations = [None for _ in range(self.n_layers)]  # type: List[t.Tensor | None]

            self.errors = [None for _ in range(self.n_layers)]  # type: List[t.Tensor | None]

            self.weights = [
                t.nn.init.xavier_normal_(t.empty(from_, to))
                for [from_, to] in zip(shape, shape[1:])
            ]

            self.activation_fn = t.sigmoid
            self.activation_fn_gradient = sigmoid_gradient

    # @jaxtyped(typechecker=typechecker)
    # def error_prediction(
    #     x0: Float[t.Tensor, "batch act0"],
    #     x1: Float[t.Tensor, "batch act1"],
    #     w: Float[t.Tensor, "act0 act1"],
    #     f,
    # ) -> Float[t.Tensor, "batch act1"]:
    #     # Eq 11
    #     return x1 - w.matmul(f(x0))

    # @jaxtyped(typechecker=typechecker)
    # def weight_dynamics_step(
    #     e0: Float[t.Tensor, "batch act0"],
    #     e1: Float[t.Tensor, "batch act1"],
    #     x0: Float[t.Tensor, "batch act0"],
    #     w: Float[t.Tensor, "act0 act1"],
    #     df,
    # ) -> Float[t.Tensor, "batch act0"]:
    #     return -e0 + t.dot(df(x0), t.matmul(w, e1))

    def relaxation_step(self):
        x = self.activations
        e = self.errors
        w = self.weights
        f = self.activation_fn
        df = self.activation_fn_gradient

        for i in range(0, self.n_layers - 1):
            # Eq 11
            e[i + 1] = x[i + 1] - t.einsum("bn,nm->bm", f(x[i]), w[i])

        for i in range(1, self.n_layers - 1):
            # Eq 12
            dx = -e[i] + t.einsum(
                "bi,bi->bi",
                df(x[i]),
                t.einsum("nm,bm->bn", w[i], e[i + 1]),
            )
            x[i] = x[i] + self.cfg.integration_step * dx

    def weight_update(self):
        x = self.activations
        e = self.errors
        w = self.weights
        f = self.activation_fn

        for i in range(0, len(x) - 1):
            dw = t.einsum("bm,bn->bnm", e[i + 1], f(x[i])).mean(dim=0)
            # dw_mag = (dw * dw).mean(0).mean(0).cpu().item()
            w[i] = w[i] + self.cfg.learning_rate * dw

    def clear_activaitons(self, batch: int):
        x = self.activations
        with self.device:
            for i in range(self.n_layers):
                x[i] = t.zeros(batch, self.shape[i])

    def forward(self, input_: t.Tensor):
        self.clear_activaitons(input_.shape[0])
        self.activations[0] = input_.to(self.device)

        for _ in range(self.cfg.n_relaxation_steps):
            self.relaxation_step()

        return t.matmul(self.activations[-2], self.weights[-1])

    def training_step(self, input_: t.Tensor, output: t.Tensor, n_relaxations_steps=32):
        self.clear_activaitons(input_.shape[0])
        input_ = input_.to(self.device)
        output = output.to(self.device)
        self.activations[0] = input_
        self.activations[-1] = output

        for _ in range(self.cfg.n_relaxation_steps):
            self.relaxation_step()

        loss = F.mse_loss(t.matmul((self.activations[-2]), self.weights[-1]), output)

        self.weight_update()

        return loss


# %%
model = PCN(PCNConfig(), [1, 1, 2])

for i in range(1000):
    loss = model.training_step(
        t.tensor([[1]]).to("cuda"), t.tensor([[0, 1]]).to("cuda")
    )
    ic(i, loss)

# %%
model.forward(t.tensor([[1]]).to("cuda"))
# %%
from torch.utils.data import DataLoader

# Define transformations for normalizing images
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # Mean and standard deviation of MNIST
    ]
)

# Load training and test sets
train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("data", train=False, download=True, transform=transform)

# Create data loaders for batching
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=True)

# %%
model = PCN(PCNConfig(), [784, 33, 16, 10])


for [i, [input_, label]] in enumerate(train_loader):
    print(f"\n step {i}")

    input_ = input_.flatten(-3, -1).to(model.device)
    label = F.one_hot(label, num_classes=10).to(model.device)
    loss = model.training_step(input_, label)

    print("loss", loss)
    # print("e[-2 norm]", model.errors[-1].norm())
    # print("x[-2] norm", model.activations[-2].norm())
    # print("eigenvalue", t.svd(model.weights[-1])[1])

    if i > 10000:
        break
