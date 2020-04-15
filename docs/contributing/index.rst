Contributing to Olympus
=======================

Adding new Basic Blocks
-----------------------

Models, Optimizers, LRSchedules, Datasets all use factories.
To insert them you simply need to create a new file inside their respective modeule.
``olympus/models/..`` for Models and register the model constructor.

Models
~~~~~~

Create a new ``olympus/models/<my_model>.py``

See :ref:`custom-model-example`

.. code-block:: python

    import torch.nn as nn

    class MyCustomModel(nn.Module):
        def __init__(self, input_size, output_size):
            self.main = nn.Linear(input_size[0], output_size[0])

        def forward(self, x):
            return self.main(x)

    # Register my model
    builders = {'my_model': MyCustomModel}


Model Optimizer
~~~~~~~~~~~~~~~

Create a new ``olympus/optimizers/<my_optimizer>.py``

See :ref:`custom-optimizer-example`

.. code-block:: python

    import torch.optim as optim

    class MyCustomOptimizer(optim.Optimizer):
        pass

    # Register my Optimizer
    builders = {'my_optimizer': MyCustomOptimizer}

Weight Initialization
~~~~~~~~~~~~~~~~~~~~~

Create a new ``olympus/models/inits/<my_init>.py``


Schedule
~~~~~~~~

See :ref:`custom-schedule-example`

Create a new ``olympus/optimizers/schedules/<my_optimizer>.py``

Tasks
~~~~~

Task describe generic setup like classification

Create a new ``olympus/tasks/<my_task>.py``

Baselines
~~~~~~~~~

Baselines are the top level scripts used to run a given tasks

Create a new ``olympus/baselines/<my_baseline>.py``

Datasets
~~~~~~~~

Create a new ``olympus/datasets/<my_baseline>.py``

Dataset Sampling
~~~~~~~~~~~~~~~~

Create a new ``olympus/datasets/sampling/<my_sampler>.py``

Metrics
~~~~~~~

Create a new ``olympus/metrics/<my_metric>.py``


Observers
~~~~~~~~~

Create a new ``olympus/observers/<my_observer>.py``

Hyper-parameter Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new ``olympus/hpo/<my_hpo>.py``


Specifying hyper-parameters
---------------------------

To add new hyper parameters you simply need to override the static method ``get_space()``

See :ref:`custom-model-nas-example`


Examples
--------

.. toctree::
   extending/custom_model
   extending/custom_model_nas
   extending/custom_optimizer
   extending/custom_schedule
   extending/custom_observer
