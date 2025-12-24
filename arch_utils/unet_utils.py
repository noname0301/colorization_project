
import torch
import torch.nn as nn

class UnetBlock(nn.Module):
    def __init__(self, up_in_c, skip_in_c, n_out):
        super().__init__()
        self.shuf = nn.Sequential(
            nn.Conv2d(up_in_c, n_out * 4, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True)
        )
        self.bn = nn.BatchNorm2d(skip_in_c)
        self.conv = nn.Conv2d(n_out + skip_in_c, n_out, 3, padding=1)

    def forward(self, up_in, skip):
        up_out = self.shuf(up_in)
        cat_x = torch.cat([up_out, self.bn(skip)], dim=1)
        return self.conv(cat_x)