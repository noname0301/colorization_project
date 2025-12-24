import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    # Simple self-attention block
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch, C, -1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, width, height)
        out = self.gamma * out + x
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, nf=64, n_blocks=3):
        super().__init__()

        self.layers = nn.ModuleList()

        # First conv
        self.layers.append(nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        current_nf = nf

        for i in range(n_blocks):
            # Conv keeping same channels
            self.layers.append(nn.Conv2d(current_nf, current_nf, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))

            # Conv downsampling, double channels
            self.layers.append(nn.Conv2d(current_nf, current_nf*2, kernel_size=4, stride=2, padding=1))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))

            # Optional self-attention on first block
            if i == 0:
                self.layers.append(SelfAttention(current_nf*2))

            current_nf *= 2

        # Final convs
        self.layers.append(nn.Conv2d(current_nf, current_nf, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final output conv: 1-channel
        self.layers.append(nn.Conv2d(current_nf, 1, kernel_size=4, stride=1, padding=0))

        # Combine as sequential for forward pass
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.model(x)
        return out.view(out.size(0), -1)