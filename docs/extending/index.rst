Extending Olympus
=================

Adding Model, Optimizer, Dataset, LRSchedule
--------------------------------------------

You can use custom classes with Olympus. In fact most of Olympus classes are also pytorch classes in disguise.
``olympus.models.Model`` is a ``torch.nn.Module``, ``olympus.optimizers.Optimizer`` is a ``torch.optim.Optimizer`` etc...
Which means Olympus code and PyTorch code are both compatible and can compose together nicely.


.. code-block:: python

    import torch.nn as nn

    class MyCustomModel(nn.Module):
        def __init__(self, input_size, output_size):
            self.main = nn.Linear(input_size[0], output_size[0])

        def forward(self, x):
            return self.main(x)

Additionally, most wrappers allow users to provide an override instead of the included models.

.. code-block:: python

    from olympus.models import Model

    model = Model(
        model=MyCustomModel,
        input_size=(290,),
        output_size=(10,)
    )

Custom builder can also be registered using ``register_<entity>``, if you rather not instantiate

.. code-block:: python

    from olympus.models import Model, register_model

    register_model('my_model', MyCustomModel)

    model = Model(
        'my_model',
        input_size=(290,),
        output_size=(10,)
    )


Creating New Metrics
--------------------

.. code-block:: python

    from dataclass import dataclass
    from olympus.metrics import Metric

    @dataclass
    class ProgressPrinter(Metric):
        frequency_epoch: int = 1        # run every epoch
        frequency_batch: int = 100      # run every 100 batch

        def on_new_batch(self, step, task=None, input=None, context=None):
            print('step', step)

        def on_new_epoch(self, epoch, task=None, context=None)
            print('epoch', epoch)

