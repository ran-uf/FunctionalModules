import torch


def gaussian(x, y, sigma=1):
    return torch.exp(-sigma * torch.sum((x - y) ** 2, dim=-1))


def batch_gaussian(x, y, sigma=1):
    return gaussian(x.transpose(0, -2), y, sigma).transpose(0, -1)


class KParameter(torch.nn.Parameter):
    def __init__(self, par):
        # super(KParameter, self).__init__()
        torch.nn.Parameter.__init__(par)


class KLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out, sigma):
        super(KLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.sigma = sigma
        self.register_parameter('w', KParameter(torch.zeros(1, dim_in + dim_out)))
        self.forward_entries = []
        self.backward_entries = []
        self.register_forward_pre_hook(self.forward_pre_hook)
        self.register_full_backward_hook(self.backward_hook)

    def _step(self):
        forward_entries = torch.stack(self.forward_entries, dim=1).reshape(-1, self.dim_in)
        backward_entries = torch.stack(self.backward_entries[::-1], dim=1).reshape(-1, self.dim_out)
        self.w.data = torch.cat([torch.cat([forward_entries, -20 * backward_entries], dim=1), self.w.data], dim=0)

    def _zero_grad(self):
        self.forward_entries = []
        self.backward_entries = []

    def forward_pre_hook(self, module, inp):
        # print(module, inp)
        self.forward_entries.append(inp[0])

    def backward_hook(self, module, p1, p2):
        self.backward_entries.append(p2[0])

    @staticmethod
    def batch_gaussian(w, x, sigma):
        return gaussian(w.unsqueeze(0), x.unsqueeze(1), sigma)

    def forward(self, x):
        return self.batch_gaussian(self.w[:, :self.dim_in], x, sigma=self.sigma) @ self.w[:, self.dim_in:]


class KernelRNNCell(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma):
        super(KernelRNNCell, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.linear = KLinear(input_dim + hidden_dim, hidden_dim, sigma)

    def forward(self, x, state):
        if state is None:
            state = torch.zeros(x.shape[0], self.hidden_size)
        o = torch.tanh(self.linear(torch.cat([x, state], dim=1)))
        return o, o


class KernelRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sigma):
        super(KernelRNN, self).__init__()
        self.cell = KernelRNNCell(input_dim, hidden_dim, sigma)
        self.o = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, state=None):
        o = []
        for i in range(x.shape[1]):
            _o, state = self.cell(x[:, i], state)
            o.append(_o)
        return self.o(torch.stack(o, dim=1)), state


model = KernelRNN(8, 12, 4, 1)
inputs = torch.randn(1, 64, 8)
target = torch.randn(1, 64, 4)
criterion = torch.nn.MSELoss()
for i in range(10000):
    model.cell.linear._zero_grad()
    pred, _ = model(inputs)
    loss = criterion(pred, target)
    loss.backward()
    print(loss.item())
    model.cell.linear._step()
