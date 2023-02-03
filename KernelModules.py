from KernelLayers import *


class KernelRNNCell(KernelModule):
    def __init__(self, input_dim, hidden_dim, sigma):
        super(KernelRNNCell, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.kernel_linear = KernelLinear(input_dim + hidden_dim, hidden_dim, sigma)

    def forward(self, x, state):
        if state is None:
            state = torch.zeros(x.shape[0], self.hidden_size)
        o = torch.tanh(self.kernel_linear(torch.cat([x, state], dim=1)))
        return o, o


class KernelRNN(KernelModule):
    def __init__(self, input_dim, hidden_dim, output_dim, sigma):
        super(KernelRNN, self).__init__()
        self.cell = KernelRNNCell(input_dim, hidden_dim, sigma)
        self.o = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        # self.o.weight.data = torch.zeros(self.o.weight.data.shape)
        # self.o.weight.data[:, -1] = 1
        self.ini_hidden = torch.randn(1, hidden_dim)

    def forward(self, x, state=None):
        o = []
        if not state:
            state = self.ini_hidden.repeat(x.shape[0], 1)
        for i in range(x.shape[1]):
            _o, state = self.cell(x[:, i], state)
            o.append(_o)
        return self.o(torch.stack(o, dim=1)), state