"""U-Net: (mask, signed distance, log Re) -> time-averaged (ux, uy, Cp), standardised."""
import torch
from torch import nn

from ..params import FIELD_TRAIN, FieldTrainParams


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class FieldUNet(nn.Module):
    def __init__(self, params: FieldTrainParams = FIELD_TRAIN, c_in: int = 3, c_out: int = 3):
        super().__init__()
        widths = [params.base_channels * 2 ** i for i in range(params.levels + 1)]
        self.down = nn.ModuleList()
        c = c_in
        for w in widths[:-1]:
            self.down.append(_block(c, w))
            c = w
        self.pool = nn.MaxPool2d(2)
        self.bottom = _block(c, widths[-1])
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for w_skip, w_below in zip(reversed(widths[:-1]), reversed(widths[1:])):
            self.up.append(nn.ConvTranspose2d(w_below, w_skip, 2, stride=2))
            self.dec.append(_block(2 * w_skip, w_skip))
        self.head = nn.Conv2d(widths[0], c_out, 1)
        self.factor = 2 ** params.levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) with H and W divisible by 2**levels. Returns (B, 3, H, W)."""
        if x.shape[-1] % self.factor or x.shape[-2] % self.factor:
            raise ValueError(f"input size {tuple(x.shape[-2:])} not divisible by {self.factor}")
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
