import torch


# class _ParameterMeta(torch._C._TensorMeta):
#     # Make `isinstance(t, Parameter)` return True for custom tensor instances that have the _is_param flag.
#     def __instancecheck__(self, instance):
#         return super().__instancecheck__(instance) or (
#             isinstance(instance, torch.Tensor) and getattr(instance, '_is_param', False))
#
#
# class KernelTensor(torch.Tensor):
#     @staticmethod
#     def __new__(cls, x, in_dim, out_dim, *args, **kwargs):
#         return super().__new__(cls, x, *args, **kwargs)
#
#     def __init__(self, x, dim_in, dim_out):
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         self.entries_forward = []
#         self.entries_backward = []


class KernelParameter(torch.nn.Parameter):
    pass


class KernelLayerBase(torch.nn.Module):
    pass


class KernelModule(torch.nn.Module):
    def kernel_parameters(self):
        parameters = torch.nn.ModuleList()
        for m in self.modules():
            if isinstance(m, KernelLayerBase):
                parameters.append(m)
        return parameters

    def parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if not isinstance(param, KernelParameter):
                yield param


def gaussian(x, y, sigma=1):
    return torch.exp(-sigma * torch.sum((x - y) ** 2, dim=-1))


def batch_gaussian(x, y, sigma=1):
    return gaussian(x.transpose(0, -2), y, sigma).transpose(0, -1)


class KernelLinear(KernelLayerBase):
    def __init__(self, in_features, out_features, sigma=1):
        super(KernelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.register_parameter('w', KernelParameter(torch.zeros(1, in_features + out_features)))
        self.forward_entries = []
        self.backward_entries = []
        self.register_forward_pre_hook(self._forward_pre_hook)
        self.register_full_backward_hook(self._backward_hook)

    def extra_repr(self) -> str:
        return 'num_kernels={}, in_features={}, out_features={}'.format(
            self.w.shape[0], self.in_features, self.out_features
        )

    def _forward_pre_hook(self, module, inp):
        # print(module, inp)
        self.forward_entries.append(inp[0])

    def _backward_hook(self, module, p1, p2):
        self.backward_entries.append(p2[0])

    @staticmethod
    def batch_gaussian(w, x, sigma):
        return gaussian(w.unsqueeze(0), x.unsqueeze(1), sigma)

    def forward(self, x):
        return self.batch_gaussian(self.w[:, :self.in_features], x, sigma=self.sigma) @ self.w[:, self.in_features:]


class KernelConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', device=None, dtype=None, sigma=1):
        super(KernelConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.sigma = sigma

        self.unfold = torch.nn.Unfold(self.kernel_size, dilation, padding, stride)
        self.kernel_linear = KernelLinear(kernel_size[0] * kernel_size[1] * self.in_channels, self.out_channels, self.sigma)

    def forward(self, x):
        batch, _, w, h = x.shape
        x = self.unfold(x)
        features = x.shape[1]
        x = self.kernel_linear(x.transpose(1, 2).reshape(-1, features)).reshape(batch, -1, self.out_channels).transpose(1, 2)
        return x.view(batch, x.shape[1],
                      int((w + 2 * self.padding - self.dilation * (self.kernel_size[0] - 1) - 1) / self.stride + 1),
                      int((w + 2 * self.padding - self.dilation * (self.kernel_size[1] - 1) - 1) / self.stride + 1))
