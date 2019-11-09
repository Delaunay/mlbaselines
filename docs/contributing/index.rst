Contributing to Olympus
=======================

Basic Blocks
------------

Models, Optimizers, LRSchedules, Datasets all use factories.
To insert them you simply need to create a new file inside their respective modeule.
``olympus/models/..`` for Moldes and register the model constructor


.. code-block:: python

    import torch.nn as nn

    class MyCustomModel(nn.Module):
        def __init__(self, input_size, output_size):
            self.main = nn.Linear(input_size[0], output_size[0])

        def forward(self, x):
            return self.main(x)

    builders = {'my_model': MyCustomModel}



Tasks
-----

Create a new ``olympus/tasks/<my_task>.py``


Baselines
---------

Create a new ``olympus/scripts/<my_baseline>.py``
