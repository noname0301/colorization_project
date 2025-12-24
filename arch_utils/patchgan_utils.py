import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        # simple 4-layer PatchGAN
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),  # 32x32 -> 16x16
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*4, 1, 4, 1, 1)  # final output
        )
        
    def forward(self, x):
        return self.model(x)