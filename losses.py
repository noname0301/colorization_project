import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        return self.loss_weight * F.l1_loss(pred, target, weight=weight, reduction=self.reduction)
    


class PerceptualLoss(nn.Module):
    def __init__(self, loss_weight=1.0, layer_weights=None, use_input_norm=True):
        super().__init__()

        self.loss_weight = loss_weight

        if layer_weights is None:
            layer_weights = {
                'conv1_1': 0.0625,
                'conv2_1': 0.125,
                'conv3_1': 0.25,
                'conv4_1': 0.5,
                'conv5_1': 1.0,
            }
        self.layer_weights = layer_weights
        
        self.layer_ids = {
            'conv1_1': 0,
            'conv2_1': 5,
            'conv3_1': 10,
            'conv4_1': 17,
            'conv5_1': 24
        }

        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg.eval()
        self.vgg.requires_grad_(False)

        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    
    def forward(self, pred, target):
        if self.use_input_norm:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std

        loss = 0.0
        features = self.vgg.features
    
        for i, layer in enumerate(features):
            pred = layer(pred)
            target = layer(target)

            for name, id in self.layer_ids.items():
                if id == i:
                    loss += self.layer_weights[name] * F.l1_loss(pred, target)

        return self.loss_weight * loss
        

class GANLoss(nn.Module):
    def __init__(self, loss_weight=1.0, real_label_val=1.0, fake_label_val=0.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_label(self, input, target_is_real):
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        target = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target)
        return loss if is_disc else loss * self.loss_weight


class ColorfulnessLoss(nn.Module):
    """Colorfulness loss.

    Args:
        loss_weight (float): Loss weight for Colorfulness loss. Default: 1.0.

    """

    def __init__(self, loss_weight=1.0):
        super().__init__()

        self.loss_weight = loss_weight

    def forward(self, pred, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
        """
        colorfulness_loss = 0
        for i in range(pred.shape[0]):
            (R, G, B) = pred[i][0], pred[i][1], pred[i][2]
            rg = torch.abs(R - G)
            yb = torch.abs(0.5 * (R+G) - B)
            (rbMean, rbStd) = (torch.mean(rg), torch.std(rg))
            (ybMean, ybStd) = (torch.mean(yb), torch.std(yb))
            stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
            meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))
            colorfulness = stdRoot + (0.3 * meanRoot)
            colorfulness_loss += (1 - colorfulness)
        return self.loss_weight * colorfulness_loss

if __name__ == "__main__":
    pass