import torch
from torchvision.transforms import ToTensor, ToPILImage

from olympus.transforms.stn import SpatialTransformerNetwork
from olympus.transforms import Preprocessor, BatchedTransform, DimSelect


def test_simple_transforms():
    shape = 3, 32, 32

    p1 = Preprocessor(
        BatchedTransform(ToPILImage()),
        BatchedTransform(ToTensor()),
        SpatialTransformerNetwork(input_shape=shape))

    p2 = Preprocessor()

    input = torch.randn((2,) + shape)

    print(input.shape)
    out = p2(input)
    print(out.shape)

    print(input.shape)
    out = p1(input)
    print(out.shape)


def test_specific_dim_transforms():
    shape = 3, 20, 20

    p1 = Preprocessor(
        BatchedTransform(ToPILImage(), dim=0),
        BatchedTransform(ToTensor(), dim=0),
        DimSelect(
            SpatialTransformerNetwork(input_shape=shape),
            dim=0)
    )

    p2 = Preprocessor()

    input = torch.randn((2,) + shape)
    target = torch.randn((2,) + (1,))

    print(len(input))
    out = p2((input, target))
    print(len(out))

    print(len(input))
    out = p1((input, target))
    print(len(out))


if __name__ == '__main__':
    test_simple_transforms()
    test_specific_dim_transforms()
