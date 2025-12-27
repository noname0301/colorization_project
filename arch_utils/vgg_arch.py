import os
import torch
from collections import OrderedDict
from torch import nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

vgg16_bn_names = [
    'conv1_1', 'bn1_1', 'relu1_1',
    'conv1_2', 'bn1_2', 'relu1_2',
    'pool1',
    'conv2_1', 'bn2_1', 'relu2_1',
    'conv2_2', 'bn2_2', 'relu2_2',
    'pool2',
    'conv3_1', 'bn3_1', 'relu3_1',
    'conv3_2', 'bn3_2', 'relu3_2',
    'conv3_3', 'bn3_3', 'relu3_3',
    'pool3',
    'conv4_1', 'bn4_1', 'relu4_1',
    'conv4_2', 'bn4_2', 'relu4_2',
    'conv4_3', 'bn4_3', 'relu4_3',
    'pool4',
    'conv5_1', 'bn5_1', 'relu5_1',
    'conv5_2', 'bn5_2', 'relu5_2',
    'conv5_3', 'bn5_3', 'relu5_3',
    'pool5'
]


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self,
                 layer_name_list,
                 use_input_norm=True,
                 requires_grad=False):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.names = vgg16_bn_names

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        # use vgg16_bn
        vgg_net = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)

        features = vgg_net.features[:max_idx + 1]

        self.vgg_net = nn.Sequential(OrderedDict(zip(self.names, features)))

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output


if __name__ == '__main__':
    model = VGGFeatureExtractor(['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], use_input_norm=True, requires_grad=False)
    for key, layer in model.vgg_net._modules.items():
        print(key)

    