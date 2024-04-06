# %%
import torch as t
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F


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
    def __init__(self, model: nn.Sequential, writer: SummaryWriter):
        super().__init__()
        self.inner = model
        self.chunks, self.pc_layers = self._build_chunks()
        self.step = 0
        self.writer = writer

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
                self.writer.add_scalar(f"dE/{i}", dE.item(), self.step)
                energy = energy + dE
                x = module.activation
            else:
                x = module(x)

        loss = loss_fn(x, labels)

        self.writer.add_scalar("energy", energy.item(), self.step)
        self.writer.add_scalar("loss", loss.item(), self.step)

        return energy, loss
    
    def _global_energy(self, x: t.Tensor, loss_fn, labels: t.Tensor) -> t.Tensor:
        energy = t.tensor(0.0, requires_grad=True)
        acts = [x] + [l.activation for l in self.pc_layers] + [labels]
        for i, [prev, curr, chunk] in enumerate(zip(acts[:-1], acts[1:], self.chunks)):
            if i == len(self.chunks) - 1:
                error = loss_fn(chunk(prev), curr)
            else:
                error = F.mse_loss(chunk(prev), curr)
            energy = energy + error
        
        return energy


    def _build_chunks(self):
        chunks = nn.ModuleList()
        pc_layers = nn.ModuleList()

        curr_chunk = nn.Sequential()
        for module in self.inner.children():
            if isinstance(module, PCLayer):
                chunks.append(curr_chunk)
                curr_chunk = nn.Sequential()
                pc_layers.append(module)
            else:
                curr_chunk.append(module)

        chunks.append(curr_chunk)

        return chunks, pc_layers
            