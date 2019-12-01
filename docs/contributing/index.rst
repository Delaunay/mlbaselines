Contributing to Olympus
=======================

Basic Blocks
------------

Models, Optimizers, LRSchedules, Datasets all use factories.
To insert them you simply need to create a new file inside their respective modeule.
``olympus/models/..`` for Moldes and register the model constructor


Models
~~~~~~

Create a new ``olympus/models/<my_model>.py``

.. code-block:: python

    import torch.nn as nn

    class MyCustomModel(nn.Module):
        def __init__(self, input_size, output_size):
            self.main = nn.Linear(input_size[0], output_size[0])

        def forward(self, x):
            return self.main(x)

    builders = {'my_model': MyCustomModel}


Initialization
~~~~~~~~~~~~~~

Create a new ``olympus/models/inits/<my_init>.py``


Optimizer
~~~~~~~~~

Create a new ``olympus/optimizers/<my_optimizer>.py``

.. code-block:: python

    import torch.optim as optim

    class MyCustomOptimizer(optim.Optimizer):
        pass

    builders = {'my_optimizer': MyCustomOptimizer}

Schedule
~~~~~~~~

Create a new ``olympus/optimizers/schedules/<my_optimizer>.py``

Tasks
-----

Create a new ``olympus/tasks/<my_task>.py``

Baselines
---------

Create a new ``olympus/baselines/<my_baseline>.py``

Datasets
--------

Create a new ``olympus/datasets/<my_baseline>.py``

Sampling
--------

Create a new ``olympus/datasets/sampling/<my_sampler>.py``

Metrics
--------

Create a new ``olympus/metrics/<my_metric>.py``
