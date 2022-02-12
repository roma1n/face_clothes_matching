import torch
from torch import nn


class DSSM(nn.Module):
    def __init__(self, x_arch, y_arch):
        super().__init__()

        self.x_arch = x_arch
        self.y_arch = y_arch

        assert len(self.x_arch) >= 2 and len(self.y_arch) >= 2, 'Expected at least one fc layer'

        self.x_embedder = nn.Sequential(*[
            nn.Linear(in_size, out_size) for in_size, out_size in zip(self.x_arch[:-1], self.x_arch[1:])
        ])

        self.y_embedder = nn.Sequential(*[
            nn.Linear(in_size, out_size) for in_size, out_size in zip(self.y_arch[:-1], self.y_arch[1:])
        ])

    def forward(self, x, y):
        x_embedding = self.x_embedder(x)
        y_embedding = self.y_embedder(y)
        return torch.sum(x_embedding * y_embedding, dim=1)


def main():
    pass


if __name__ == '__main__':
    from torchinfo import summary

    dssm = DSSM(
        x_arch=[512, 128, 32],
        y_arch=[512, 128, 32],
    )
    batch_size = 100

    summary(dssm, ((batch_size, 512), (batch_size, 512)))
