"""CNN: obstacle mask -> conv features; standardised log(Re) joins in the dense head."""
import torch
from torch import nn

from ..params import TRAIN, TrainParams


class SurrogateCNN(nn.Module):
    def __init__(self, params: TrainParams = TRAIN, n_out: int = 2):
        super().__init__()
        layers, c_in = [], 1
        for c in params.channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(c_in + 1, params.hidden), nn.ReLU(inplace=True),
                                  nn.Linear(params.hidden, n_out))

    def forward(self, mask: torch.Tensor, re: torch.Tensor) -> torch.Tensor:
        """mask: (B, 1, H, W) float in {0, 1}; re: (B, 1) standardised log(Re)."""
        return self.head(torch.cat([self.features(mask), re], dim=1))
