import torch
from torch import nn


class ExtendedResnetEmbedder(nn.Module):
    def __init__(
        self,
        pretrained=True,
        n_extend_chennels=1,
        extention_alpha=1e-2,
    ):
        super().__init__()

        self.pretrained = pretrained
        self.n_extend_chennels = n_extend_chennels
        self.extention_alpha = extention_alpha

        self.base = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=self.pretrained)

        if self.n_extend_chennels > 0:
            self._extend_base(self.n_extend_chennels)

        self.base.fc = nn.Identity()


    def _extend_base(self, channels):
        self.base.conv1.weight.data = torch.cat(
            [
                self.base.conv1.weight.data,
                torch.normal(0, self.extention_alpha, size=(64, 1, 7, 7)),
            ], 
            dim=1,
        )

    def forward(self, x):
        x = x[:, :3 + self.n_extend_chennels]
        return self.base.forward(x)


def main():
    from torchinfo import summary

    net = ExtendedResnetEmbedder()
    batch_size = 100
    summary(net, (batch_size, 4, 224, 224))


if __name__ == '__main__':
    main()
