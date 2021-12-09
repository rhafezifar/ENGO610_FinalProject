import numpy as np
import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict

import torchvision
from matplotlib import pyplot as plt

from plot_utils import plot_loss, plot_output
from train_model import train_model
from test_model import test_model
from dataLoader import classes, test_loader, batch_size, reverse_transform


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=c, out_features=c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=c // r, out_features=c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)

            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm2d(out_channels)}))


class SEResNetBasicBlock(ResNetResidualBlock):
    # Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
            # adding SE block
            SE_Block(self.expanded_channels)
        )


class SEResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation(),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
            # adding SE block
            SE_Block(self.expanded_channels)
        )


class SEResNetLayer(nn.Module):
    # A ResNet layer composed by `n` blocks stacked one after the other

    def __init__(self, in_channels, out_channels, block=SEResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class SEResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2, 2, 2, 2],
                 activation=nn.ReLU, block=SEResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]), activation(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            SEResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, block=block, *args, **kwargs),
            *[
                SEResNetLayer(in_channels * block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs)
                for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])
            ]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = SEResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def seresnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=SEResNetBasicBlock, deepths=[2, 2, 2, 2])


def seresnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=SEResNetBasicBlock, deepths=[3, 4, 6, 3])


def seresnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=SEResNetBottleNeckBlock, deepths=[3, 4, 6, 3])


def seresnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=SEResNetBottleNeckBlock, deepths=[3, 4, 23, 3])


def seresnet152(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=SEResNetBottleNeckBlock, deepths=[3, 8, 36, 3])


if __name__ == '__main__':
    # from torchsummary import summary

    model = seresnet34(3, len(classes))
    # summary(model.cuda(), (3, 256, 256))

    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 20

    model, train_loss, valid_loss = train_model(model, patience, 100)
    if train_loss and valid_loss:
        plot_loss(train_loss, valid_loss)

    ###############################################################################################################################
    test_model(model)

    # obtain one batch of test images
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.to(gpu)
    labels = labels.to(gpu)
    # get sample outputs
    output = model(images)
    images = images.cpu()
    # convert output probabilities to predicted class
    _, preds = torch.max(output, 1)

    plot_output(images, labels, preds)
