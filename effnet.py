import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(in_, out_, stride=1, padding=0, **conv_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_, out_, stride=stride, padding=padding, **conv_kwargs),
        nn.BatchNorm2d(out_),
        nn.ReLU(True)
    )


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SEModule(nn.Module):
    def __init__(self, in_, ratio=0.25):
        super().__init__()
        mid_ = int(in_ * ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            *conv_bn_act(in_, mid_, kernel_size=1, bias=False),
            nn.Conv2d(mid_, in_, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))


class DropConnect(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        if not self.training():
            return x

        random_tensor = self.ratio
        random_tensor += torch.rand([x.shape[0], 1, 1, 1], dtype=torch.float)
        return x / self.ratio * random_tensor


class MBConv(nn.Module):
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip,
                 se_ratio, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = conv_bn_act(in_, mid_, kernel_size=1, bias=False) if expand != 1 else nn.Identity()

        self.dw_conv1 = conv_bn_act(mid_, mid_, 
                                    kernel_size=kernel_size, stride=stride, padding=kernel_size//2,
                                    groups=mid_, bias=False)

        self.se = SEModule(mid_, se_ratio) if se_ratio > 0 else nn.Identity()
        # ProjectConv : kernel 1x1
        self.project_conv = conv_bn_act(mid_, out_, kernel_size=1, bias=False)

        # if _block_args.id_skip:
        # and all(s == 1 for s in self._block_args.strides)
        # and self._block_args.input_filters == self._block_args.output_filters:
        self.skip = skip and (stride == 1) and (in_ == out) 
        if self.skip:
            self.dropconnect = DropConnect(dc_ratio)

    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.dw_conv1(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = x + self.dropconnect(inputs)
        return x


class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio, drop_connect_ratio)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip, se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet(nn.Module):
    def __init__(self, width_coeff, depth_coeff,
                 depth_div=8, min_depth=None,
                 dropout_rate=0.2, drop_connect_rate=0.2,
                 num_classes=1000):
        super().__init__()
        min_depth = min_depth or depth_div

        self.stem = conv_bn_act(3, 32, kernel_size=3, stride=2, padding=1, bias=False)

        def renew_ch(x):
            if not width_coeff:
                return x

            new_x = x * width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * new_x:
                new_x += depth_div
            return new_x

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        self.blocks = nn.Sequential(
            MBBlock(renew_ch(32),  renew_ch(16),  1, 3, 1, renew_repeat(1), False, 0.25, drop_connect_rate),
            MBBlock(renew_ch(16),  renew_ch(24),  6, 3, 2, renew_repeat(2), False, 0.25, drop_connect_rate),
            MBBlock(renew_ch(24),  renew_ch(40),  6, 5, 2, renew_repeat(2), False, 0.25, drop_connect_rate),
            MBBlock(renew_ch(40),  renew_ch(80),  6, 3, 2, renew_repeat(3), False, 0.25, drop_connect_rate),
            MBBlock(renew_ch(80),  renew_ch(112), 6, 5, 1, renew_repeat(3), False, 0.25, drop_connect_rate),
            MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4), False, 0.25, drop_connect_rate),
            MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(1), False, 0.25, drop_connect_rate)
        )

        self.head = nn.Sequential(
            *conv_bn_act(320, 1280, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout_rate, True) if dropout_rate > 0 else nn.Identity(),
            Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, inputs):
        stem = self.stem(inputs)
        x = self.blocks(stem)
        return self.head(x)


if __name__ == "__main__":
    print("Efficient B0 Summary")
    net = EfficientNet(1, 1)
    from torchsummary import summary
    summary(net.cuda(), (3, 224, 224))
