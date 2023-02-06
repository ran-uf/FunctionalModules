import torch


class MyDense(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(MyDense, self).__init__()
        self.l1 = torch.nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias)

    def forward(self, x):
        x = self.l1(x)
        w = self.l1.weight.norm(dim=1)
        return x / w.unsqueeze(0)
