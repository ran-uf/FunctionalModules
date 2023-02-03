import torch


class KernelOptimizer(object):
    def __init__(self, parameters: torch.nn.ParameterList, lr=3e-5, quantization=1e-2, *args, **kwargs):
        # super(KernelOptimizer, self).__init__(layers, kwargs)
        self.parameters = parameters
        self.quantization = quantization
        self.lr = lr

    def step(self, *arg, **kwargs):
        for parameter in self.parameters:
            self._step(parameter)

    def _step(self, parameter):
        forward_entries = torch.stack(parameter.forward_entries, dim=1).reshape(-1, parameter.in_features)
        backward_entries = torch.stack(parameter.backward_entries[::-1], dim=1).reshape(-1, parameter.out_features)
        grad = torch.cat([forward_entries, -self.lr * backward_entries], dim=1)

        def self_merge(x, split):
            sim = torch.mean((x[:, :split].unsqueeze(1) - x[:, :split].unsqueeze(0)) ** 2, dim=-1)
            sim = sim + torch.triu(torch.ones(sim.shape), diagonal=0) < self.quantization
            indices = sim.nonzero()
            if indices.shape[0] == 0:
                return x
            else:
                indices = indices.numpy().tolist()
                # print(indices)
                mp_i = []
                mp_j = []
                while indices:
                    [temp_i, temp_j] = indices.pop()
                    mp_i.append(temp_i)
                    mp_j.append(temp_j)
                    indices = [i for i in indices if i[1] != temp_i and i[0] != temp_i
                                                 and i[1] != temp_j and i[0] != temp_j]
                    # for _i in indices:
                    #     if _i[1] == temp_i or _i[0] == temp_i:
                    #         indices.remove(_i)
                for i, j in zip(mp_i, mp_j):
                    x[j, split:] += x[i, split:]
                ids = [i for i in range(x.shape[0])]
                # print(ids, mp_i)
                for i in mp_i:
                    ids.remove(i)
                return x[ids]

        # merge x to y
        def merge(x, y, split):
            sim = torch.mean((x[:, :split].unsqueeze(1) - y[:, :split].unsqueeze(0)) ** 2, dim=-1)
            sim = sim < self.quantization
            indices = sim.nonzero()
            if indices.shape[0] == 0:
                return torch.cat([x, y], dim=0)
            else:
                indices = indices.numpy().tolist()
                mp_i = []
                mp_j = []
                while indices:
                    [temp_i, temp_j] = indices.pop()
                    mp_i.append(temp_i)
                    mp_j.append(temp_j)
                    indices = [i for i in indices if i[0] != temp_i]

                for i, j in zip(mp_i, mp_j):
                    y[j, split:] += x[i, split:]
                ids = [i for i in range(x.shape[0])]
                for i in mp_i:
                    ids.remove(i)
                return torch.cat([y, x[ids]], dim=0)

        grad = self_merge(grad, parameter.in_features)
        parameter.w.data = merge(grad, parameter.w.data, parameter.in_features)
        parameter.w.data = self_merge(parameter.w.data, parameter.in_features)

    def zero_grad(self, *args, **kwargs):
        for parameter in self.parameters:
            self._zero_grad(parameter)

    @staticmethod
    def _zero_grad(parameter):
        parameter.forward_entries = []
        parameter.backward_entries = []
