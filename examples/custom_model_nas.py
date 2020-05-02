import torch
import torch.nn as nn

from olympus.models import Model


class MyCustomNASModel(nn.Module):
    def __init__(self, input_size, output_size, l1, l2, l3, l4):
        super(MyCustomNASModel, self).__init__()

        modules = []
        prev = input_size[0]
        for size in [l1, l2, l3, l4]:
            modules.append(nn.Linear(prev, size))
            prev = size

        modules.append(nn.Linear(prev, output_size[0]))
        self.main = nn.Sequential(*modules)

    def forward(self, x):
        return self.main(x)

    @staticmethod
    def get_space():
        return {
            'l1': 'uniform(32, 64, discrete=True)',
            'l2': 'uniform(32, 64, discrete=True)',
            'l3': 'uniform(32, 64, discrete=True)',
            'l4': 'uniform(32, 64, discrete=True)'
        }


# Register my model
builders = {'my_model': MyCustomNASModel}


if __name__ == '__main__':
    model = Model(
        model=MyCustomNASModel,
        input_size=(290,),
        output_size=(10,),
        # Fix this hyper-parameter right away
        l1=21
    )

    # If you use an hyper parameter optimizer, it will generate this for you
    model.init(l2=33, l3=33, l4=32)

    input = torch.randn((10, 290))
    out = model(input)

    print(out)
