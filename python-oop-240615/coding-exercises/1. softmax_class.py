import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_exp = torch.exp(x)
        total = x_exp.sum(0, keepdims=True)
        return x_exp / total


class softmax_stable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = torch.max(x, dim=0)
        x_exp = torch.exp(x - c.values)
        total = x_exp.sum(0, keepdims=True)
        return x_exp / total


data = torch.Tensor([1, 2, 3])
softmax = Softmax()
output = softmax(data)
print(output)

data = torch.Tensor([1, 2, 3])
softmax_stable = softmax_stable()
output = softmax_stable(data)
print(output)
