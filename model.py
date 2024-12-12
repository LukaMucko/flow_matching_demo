import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout else nn.Identity(),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv(x)
        return h + self.shortcut(x)


class FlowMatchingModel(nn.Module):
    def __init__(self, channels=32, num_blocks=3, dropout=0.1):
        super().__init__()

        self.init_conv = nn.Conv2d(4, channels, kernel_size=3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        for i in range(num_blocks):
            curr_channels = channels * (2**i)
            next_channels = curr_channels * 2
            self.encoder_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(curr_channels, next_channels, dropout=dropout),
                        ResidualBlock(next_channels, next_channels, dropout=dropout),
                        nn.MaxPool2d(2),
                    ]
                )
            )

        mid_channels = channels * (2**num_blocks)
        self.middle_block = nn.Sequential(
            ResidualBlock(mid_channels, mid_channels, dropout=dropout),
            ResidualBlock(mid_channels, mid_channels, dropout=dropout),
            ResidualBlock(mid_channels, mid_channels, dropout=dropout),
        )

        self.decoder_blocks = nn.ModuleList()
        for i in range(num_blocks):
            curr_channels = channels * (2 ** (num_blocks - i))
            self.decoder_blocks.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            curr_channels, curr_channels // 2, 2, stride=2
                        ),
                        ResidualBlock(
                            curr_channels + curr_channels // 2,
                            curr_channels // 2,
                            dropout=dropout,
                        ),
                        ResidualBlock(
                            curr_channels // 2, curr_channels // 2, dropout=dropout
                        ),
                    ]
                )
            )

        self.final_conv = nn.Conv2d(channels, 3, kernel_size=1)

    def forward(self, x: Tensor, t: Tensor):
        #t: Tensor shape (b, 1) | (b)
        b, c, h, w = x.shape
        t = t.view(b, 1, 1, 1).expand(b, 1, h, w)
        x = torch.cat([x, t], dim=1)

        x = self.init_conv(x)

        skip_connections = []
        for res1, res2, pool in self.encoder_blocks:
            x = res1(x)
            x = res2(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.middle_block(x)

        for i, (up, res1, res2) in enumerate(self.decoder_blocks):
            x = up(x)
            skip = skip_connections[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = res1(x)
            x = res2(x)

        return self.final_conv(x)


class FlowMLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2+1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: Tensor, t: Tensor):
        # x: (batch_size, 2), t: (batch_size,) | (b, 1)
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)