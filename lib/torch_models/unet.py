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
            self.down_scale = nn.Conv2d(channels, 2 * channels, 4, padding=1, stride=2)

        def forward(self, x):
            x = self.transform_block(x)
            x = self.down_scale(x)
            return x

    class UpScaleBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()

            self.up_scale = nn.ConvTranspose2d(4 * channels, channels, 4, padding=1, stride=2)
            self.transform_block = UNET.TransformationBlock(channels)

        def forward(self, x):
            x = self.up_scale(x)
            x = self.transform_block(x)
            return x


    def __init__(self, num_classes, base_channel_num=8):
        super().__init__()

        self.base_channel_num = base_channel_num
        self.num_classes = num_classes

        self.conv_in = nn.Conv2d(3, self.base_channel_num, 1)

        self.down_1 = UNET.DownScaleBlock(1 * self.base_channel_num)
        self.down_2 = UNET.DownScaleBlock(2 * self.base_channel_num)
        self.down_3 = UNET.DownScaleBlock(4 * self.base_channel_num)

        self.transform_block = UNET.TransformationBlock(8 * self.base_channel_num)

        self.up_3 = UNET.UpScaleBlock(4 * self.base_channel_num)
        self.up_2 = UNET.UpScaleBlock(2 * self.base_channel_num)
        self.up_1 = UNET.UpScaleBlock(1 * self.base_channel_num)

        self.conv_out = nn.Conv2d(self.base_channel_num, self.num_classes, 1)
        self.acti_out = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_in(x)

        out_1 = self.down_1(x)
        out_2 = self.down_2(out_1)
        out_3 = self.down_3(out_2)

        x = self.transform_block(out_3)

        x = self.up_3(torch.cat([x, out_3], dim=1))
        x = self.up_2(torch.cat([x, out_2], dim=1))
        x = self.up_1(torch.cat([x, out_1], dim=1))

        x = self.conv_out(x)
        x = self.acti_out(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    unet = UNET(num_classes=45)
    batch_size = 100

    summary(unet, (batch_size, 3, 224, 224))
