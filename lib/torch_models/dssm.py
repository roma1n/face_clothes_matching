import torch
from torch import nn
from torch.nn import functional as F


class DSSM(nn.Module):
    class LinearBlock(nn.Module):
        def __init__(self, in_size, out_size, p_dropout):
            super().__init__()

            self.bn = nn.BatchNorm1d(in_size)
            self.acti = nn.ReLU()
            self.dropout = nn.Dropout(p=p_dropout)
            self.fc = nn.Linear(in_size, out_size)

        def forward(self, x):
            x = self.bn(x)
            x = self.acti(x)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    def __init__(self, x_arch, y_arch, norm=False, p_dropout=0.2):
        super().__init__()

        self.x_arch = x_arch
        self.y_arch = y_arch
        self.norm = norm

        assert len(self.x_arch) >= 2 and len(self.y_arch) >= 2, 'Expected at least one fc layer'

        self.x_embedder = nn.Sequential(*[
            DSSM.LinearBlock(in_size, out_size, p_dropout) for in_size, out_size in zip(self.x_arch[:-1], self.x_arch[1:])
        ])

        self.y_embedder = nn.Sequential(*[
            DSSM.LinearBlock(in_size, out_size, p_dropout) for in_size, out_size in zip(self.y_arch[:-1], self.y_arch[1:])
        ])

    def forward(self, x, y):
        x_embedding = self.x_embedder(x)
        y_embedding = self.y_embedder(y)

        if self.norm:
            x_embedding = F.normalize(x_embedding)
            y_embedding = F.normalize(y_embedding)

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
