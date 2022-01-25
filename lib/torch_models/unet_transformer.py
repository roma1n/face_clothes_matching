import torch
from torch import nn


class UnetTransformer(nn.Module):
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

            self.conv_1 = UnetTransformer.ConvBlock(channels, channels)
            self.conv_2 = UnetTransformer.ConvBlock(channels, channels)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.conv_2(x)
            return x

    class Transformer2dBlock(nn.Module):
        def __init__(
            self,
            channels,
            num_heads=8,
            image_size=28,
        ):
            super().__init__()

            self.channels = channels
            self.num_heads = num_heads
            self.image_size = image_size
            self.length = image_size**2

            # input: batch * length x channels; output: batch * length x channels
            self.key_embedder = nn.Linear(self.channels, self.channels)
            self.query_embedder = nn.Linear(self.channels, self.channels)
            self.value_embedder = nn.Linear(self.channels, self.channels)

            self.attention = nn.MultiheadAttention(
                embed_dim=self.channels,
                num_heads=self.num_heads,
                batch_first=True,
            )

        def forward(self, x):
            batch_size = x.shape[0]
            x = x.view(batch_size, self.channels, self.length).transpose(1, 2)

            key = self.key_embedder(x)
            query = self.query_embedder(x)
            value = self.value_embedder(x)
            x, _ = self.attention(query, key, value)

            x = x.view(batch_size, self.length, self.channels).transpose(1, 2)
            x = x.view(batch_size, self.channels, self.image_size, self.image_size)

            return x

    class DownScaleBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()

            self.transform_block = UnetTransformer.TransformationBlock(channels)
            self.down_scale = nn.Conv2d(channels, 2 * channels, 4, padding=1, stride=2)

        def forward(self, x):
            x = self.transform_block(x)
            x = self.down_scale(x)
            return x

    class UpScaleBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()

            self.up_scale = nn.ConvTranspose2d(4 * channels, channels, 4, padding=1, stride=2)
            self.transform_block = UnetTransformer.TransformationBlock(channels)

        def forward(self, x):
            x = self.up_scale(x)
            x = self.transform_block(x)
            return x


    def __init__(self, num_classes, base_channel_num=8):
        super().__init__()

        self.base_channel_num = base_channel_num
        self.num_classes = num_classes

        self.conv_in = nn.Conv2d(3, self.base_channel_num, 1)

        self.down_1 = UnetTransformer.DownScaleBlock(1 * self.base_channel_num)
        self.down_2 = UnetTransformer.DownScaleBlock(2 * self.base_channel_num)
        self.down_3 = UnetTransformer.DownScaleBlock(4 * self.base_channel_num)

        self.transformer = UnetTransformer.Transformer2dBlock(8 * self.base_channel_num)

        self.up_3 = UnetTransformer.UpScaleBlock(4 * self.base_channel_num)
        self.up_2 = UnetTransformer.UpScaleBlock(2 * self.base_channel_num)
        self.up_1 = UnetTransformer.UpScaleBlock(1 * self.base_channel_num)

        self.conv_out = nn.Conv2d(self.base_channel_num, self.num_classes, 1)
        self.acti_out = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_in(x)

        out_1 = self.down_1(x)
        out_2 = self.down_2(out_1)
        out_3 = self.down_3(out_2)

        x = self.transformer(out_3)

        x = self.up_3(torch.cat([x, out_3], dim=1))
        x = self.up_2(torch.cat([x, out_2], dim=1))
        x = self.up_1(torch.cat([x, out_1], dim=1))

        x = self.conv_out(x)
        x = self.acti_out(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    unet_transformer = UnetTransformer(num_classes=45)
    batch_size = 100

    summary(unet_transformer, (batch_size, 3, 224, 224))
