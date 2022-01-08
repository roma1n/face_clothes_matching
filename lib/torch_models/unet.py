import torch
from torch import nn


class UNET(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.norm = nn.BatchNorm2d(out_channels)
            self.acti = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.norm(x)
            x = self.acti(x)
            return x

    class TransformationBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()

            self.conv_1 = UNET.ConvBlock(channels, channels)
            self.conv_2 = UNET.ConvBlock(channels, channels)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.conv_2(x)
            return x

    class DownScaleBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()

            self.transform_block = UNET.TransformationBlock(channels)
            self.down_scale = nn.Conv2d(channels, 2 * channels, 3, padding=3, stride=2)

        def forward(self, x):
            x = self.transform_block(x)
            x = self.down_scale(x)
            return x

    class UpScaleBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()

            self.transform_block = UNET.TransformationBlock(2 * channels)
            self.up_scale = nn.ConvTranspose2d(2 * channels, channels, 3, padding=3, stride=2)

        def forward(self, x):
            x = self.transform_block(x)
            x = self.up_scale(x)
            return x


    def __init__(self, base_channel_num=4):
        super().__init__()

        self.base_channel_num = base_channel_num

        self.conv_in = nn.Conv2d(3, self.base_channel_num, 1)

        self.down_1 = UNET.DownScaleBlock(1 * self.base_channel_num)
        self.down_2 = UNET.DownScaleBlock(2 * self.base_channel_num)
        self.down_3 = UNET.DownScaleBlock(4 * self.base_channel_num)

        self.up_3 = UNET.UpScaleBlock(4 * self.base_channel_num)
        self.up_2 = UNET.UpScaleBlock(2 * self.base_channel_num)
        self.up_1 = UNET.UpScaleBlock(1 * self.base_channel_num)

        self.conv_out = nn.Conv2d(self.base_channel_num, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)

        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)

        x = self.up_3(x)
        x = self.up_2(x)
        x = self.up_1(x)

        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    unet = UNET()
    summary(unet, (128, 3, 224, 224))
