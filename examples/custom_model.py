import torch
import torch.nn as nn

from olympus.models import Model


class MyCustomModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyCustomModel, self).__init__()
        self.main = nn.Linear(input_size[0], output_size[0])

    def forward(self, x):
        return self.main(x)


# Register my model
builders = {'my_model': MyCustomModel}


if __name__ == '__main__':
    model = Model(
        model=MyCustomModel,
        input_size=(290,),
        output_size=(10,)
    )

    input = torch.randn((10, 290))
    out = model(input)

    print(out)
