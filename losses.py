import math
import torch
from torch import autograd as autograd
import torch.nn as nn
from torch.nn import functional as F

from arch_utils.vgg_arch import VGGFeatureExtractor


class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return self.loss_weight * F.l1_loss(pred, target, reduction=self.reduction)


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): The perceptual loss will be calculated and the loss will multiplied by the weight.
            Default: 1.0.
    """

    def __init__(self,
                 layer_weights,
                 use_input_norm=True,
                 perceptual_weight=1.0):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            use_input_norm=use_input_norm)


    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        percep_loss = 0
        for k in x_features.keys():
            percep_loss += F.l1_loss(x_features[k], gt_features[k]) * self.layer_weights[k]
        percep_loss *= self.perceptual_weight

        return percep_loss


class VanillaGANLoss(nn.Module):
    """Define GAN loss.

    Args:
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        self.loss = nn.BCEWithLogitsLoss()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (Tensor): Target tensor.
        """
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class ColorfulnessLoss(nn.Module):
    """Colorfulness loss.

    Args:
        loss_weight (float): Loss weight for Colorfulness loss. Default: 1.0.

    """

    def __init__(self, loss_weight=1.0):
        super(ColorfulnessLoss, self).__init__()
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
