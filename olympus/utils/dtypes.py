import torch
from typing import TypeVar


class TensorMeta(type):
    def __new__(mcls, *args):
        return super().__new__(mcls, *args)

    def __getitem__(self, item):
        pass


class Tensor(metaclass=TensorMeta):
    def __init__(self, shape, device, type):
        self.shape = shape
        self.device = device
        self.type = type

    def __getitem__(self, *args):
        shape = []

        for arg in args:
            if isinstance(arg, int):
                shape.append(arg)
            elif isinstance(arg, tuple):
                shape = arg
            elif isinstance(arg, TypeVar):
                shape = arg
            elif isinstance(arg, torch.device):
                self.device = arg
            elif isinstance(arg, torch.dtype):
                self.dtype = arg
            else:
                print(f'{arg} is not recognized as a Tensor parameter')
        return self


N = TypeVar('N')
C = TypeVar('C')
H = TypeVar('H')
W = TypeVar('W')

CHW = C, H, W
NCHW = N, C, H, W

Image = Tensor[CHW]
