import torch
from torch.cuda.streams import Stream as CudaStream


class CpuStream:
    pass


def Stream(*args, **kwargs):
    if torch.cuda.is_available():
        return CudaStream()
    return None


stream = torch.cuda.stream
