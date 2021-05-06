import torch
import torch.nn as nn
import torch.nn.functional as F

class MimgnetEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(MimgnetEncoder, self).__init__()
        self.last_hidden_size = 2*2*hidden_size

        self.encoder = nn.Sequential(
            # -1, hidden_size, 42, 42
            nn.Conv2d(3, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, hidden_size, 21, 21
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, hidden_size, 10, 10
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, 2*hidden_size, 5, 5
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, hidden_size, 2, 2
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.last_hidden_size)
        return h